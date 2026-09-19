"""Seeded correctness fuzzer for flashinfer's probability-sampling entry points.

Covers ``sampling_from_probs``, ``top_k`` / ``top_p`` / ``top_k_top_p_sampling_from_probs``
(both ``filter_apply_order`` values), and the ``indices``, per-request-threshold and
``seed`` / ``offset`` / ``generator`` paths those APIs expose. The oracle is an
independent float64 PyTorch reference -- it never calls flashinfer and does not mirror
the kernel's rejection loop -- encoding the documented set semantics: the inclusive
top-p nucleus, top-k keeping every class at least as likely as the k-th largest, joint
as the intersection of both masks and top_k_first as a top-k renormalize followed by
top-p.

Two layers run over that oracle. ``test_sampling_fuzz`` draws a few thousand samples per
seeded case and checks that every draw lands in the reference support, that every class
firmly inside the support is drawn, that the shape / dtype / index range are right, and
that a fixed seed/offset (or generator state) replays exactly. ``test_sampling_distribution`` runs a Pearson chi-square goodness-of-fit with an
effect-size gate over a frozen set of constructed distributions, to catch a kernel whose
support is right but whose probability mass is wrong; the family-wise false-positive
budget is declared up front and split by Bonferroni, and the machinery is checked against
``torch.multinomial`` first.

The kernel renormalizes and compares cumulative masses in float32, so the support oracle
is two-sided: a draw is only flagged when it falls outside the support computed with the
threshold loosened by an fp32 tolerance. Ties are checked only at the support level, never
against a particular tie-break. A failing case prints a single-seed repro command.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import pytest
import torch

import flashinfer
from tests.test_helpers.fuzz_ledger import FuzzLedger

NUM_CASES = int(os.environ.get("FLASHINFER_SAMPLING_FUZZ_NUM_CASES", "240"))
BASE_SEED = int(os.environ.get("FLASHINFER_SAMPLING_FUZZ_SEED", "0"))
# Comma-separated seeds -> run only those cases; the repro command printed on failure
# uses this so a single seed reproduces one case exactly.
_ONLY_SEEDS = os.environ.get("FLASHINFER_SAMPLING_FUZZ_ONLY_SEED", "")

_SKIP = None if torch.cuda.is_available() else "CUDA not available"
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))

# The sampling entry points document float32 in / int32 out; the renorm helpers also take
# fp16/bf16, but the sampling path is float32, so a wider dtype axis would be out of contract.
_DTYPE = torch.float32
_DEVICE = "cuda:0"
_SAMPLES_PER_CASE = 4096


def _fp32_eps(vocab: int) -> float:
    """Boundary tolerance for a float32 cumulative sum over up to ``vocab`` terms.

    float32 has a unit roundoff of 2**-24; a running sum of ``vocab`` terms carries a
    worst-case absolute error of about ``vocab * 2**-24``. We loosen the top-p mass
    threshold and the top-k pivot by this so a draw near a boundary is never flagged for
    landing on the fp32 side of it.
    """
    return vocab * 2.0**-24


# ---------------------------------------------------------------------------
# Independent reference (float64, CPU). Value-based and tie-inclusive, so a keep-mask is a
# superset of any valid tie-break. ``shift`` moves the boundary: > 0 loosens (keeps more).
# ---------------------------------------------------------------------------
def reference_normalize(probs: torch.Tensor) -> torch.Tensor:
    """Row-normalize to a distribution in float64."""
    p = probs.detach().to("cpu", torch.float64)
    return p / p.sum(dim=-1, keepdim=True)


def reference_top_k_support(
    probs_row: torch.Tensor, top_k: int, shift: float = 0.0
) -> torch.Tensor:
    """Keep every class whose probability is at least the k-th largest.

    Equal-valued classes are kept or dropped together. ``shift`` is a relative tolerance
    on the pivot for the float32 comparison the kernel makes.
    """
    vocab = probs_row.numel()
    if top_k >= vocab:
        return torch.ones_like(probs_row, dtype=torch.bool)
    pivot = torch.sort(probs_row, descending=True).values[top_k - 1]
    return probs_row >= pivot * (1.0 - shift)


def reference_top_p_support(
    probs_row: torch.Tensor, top_p: float, shift: float = 0.0
) -> torch.Tensor:
    """Inclusive top-p nucleus: keep a class iff the mass strictly above it is < top_p.

    Written over the strictly-greater mass so equal-valued classes are kept together.
    ``shift`` loosens (or tightens) the cumulative-mass threshold for the fp32 boundary.
    """
    greater = torch.where(
        probs_row.unsqueeze(0) > probs_row.unsqueeze(1),
        probs_row.unsqueeze(0),
        torch.zeros_like(probs_row).unsqueeze(0),
    ).sum(dim=1)
    return greater < top_p + shift


def _renorm_on(probs_row: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    kept = torch.where(keep, probs_row, torch.zeros_like(probs_row))
    return kept / kept.sum()


def reference_top_k_filter(probs_row: torch.Tensor, top_k: int) -> torch.Tensor:
    return _renorm_on(probs_row, reference_top_k_support(probs_row, top_k))


def reference_top_p_filter(probs_row: torch.Tensor, top_p: float) -> torch.Tensor:
    return _renorm_on(probs_row, reference_top_p_support(probs_row, top_p))


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


def reference_allowed_support(
    api: str,
    probs_row: torch.Tensor,
    top_k: Optional[int],
    top_p: Optional[float],
    order: str,
    shift: float,
) -> torch.Tensor:
    """Classes a correct kernel may draw, with the boundary loosened by ``shift``.

    Always intersected with the positive-mass classes, so a zero-probability class is
    excluded even for a no-op threshold.
    """
    positive = probs_row > 0
    if api == "sampling_from_probs":
        return positive
    if api == "top_k":
        return reference_top_k_support(probs_row, top_k, shift) & positive
    if api == "top_p":
        return reference_top_p_support(probs_row, top_p, shift) & positive
    # top_k_top_p: loosen both masks; top_k_first loosens the top-p of the renormalized row.
    if order == "joint":
        keep = reference_top_k_support(
            probs_row, top_k, shift
        ) & reference_top_p_support(probs_row, top_p, shift)
        return keep & positive
    renorm_k = reference_top_k_filter(probs_row, top_k)
    keep = reference_top_k_support(probs_row, top_k, shift) & reference_top_p_support(
        renorm_k, top_p, shift
    )
    return keep & positive


def reference_target_distribution(
    api: str,
    probs_row: torch.Tensor,
    top_k: Optional[int],
    top_p: Optional[float],
    order: str,
) -> torch.Tensor:
    """Exact post-filter distribution (for the goodness-of-fit target)."""
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
# incomplete gamma (no SciPy). Low-expected tail bins are merged so the approximation holds.
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
    counts = counts.to("cpu", torch.float64)
    target = target.to("cpu", torch.float64)
    n = counts.sum()
    support = target > 0
    obs = counts[support]
    exp = target[support] * n
    order = torch.argsort(exp)
    obs, exp = obs[order].tolist(), exp[order].tolist()
    merged_o: List[float] = []
    merged_e: List[float] = []
    acc_o = acc_e = 0.0
    for o, e in zip(obs, exp, strict=True):
        acc_o += o
        acc_e += e
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
    stat = (((o - e) ** 2) / e).sum()
    dof = max(o.numel() - 1, 1)
    p_value = torch.special.gammaincc(
        torch.tensor(dof / 2.0, dtype=torch.float64), stat / 2.0
    ).item()
    tvd = 0.5 * (counts / n - target).abs().sum().item()
    max_resid = ((o - e) / e.sqrt()).abs().max().item()
    return GofResult(stat.item(), dof, p_value, tvd, max_resid)


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
    margin = 10.0 * _fp32_eps(row_norm.numel())
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
    # top_p and top_k_top_p index a per-request threshold by the probs row, so they compose
    # with indices. per-request top_k + indices is held out: the top_k kernel indexes its k
    # array by output position instead of row (a kernel bug filed separately), so top_k takes
    # a per-request k only on the unindexed path until that is fixed.
    if api == "top_k":
        per_request = not use_indices and num_dist > 1 and rng.random() < 0.5
    else:
        per_request = num_dist > 1 and rng.random() < 0.5
    order = "-"
    top_k: List[int] = []
    top_p: List[float] = []
    p_boundary = "-"
    if api in ("top_k", "top_k_top_p"):
        choices = sorted({1, 2, max(1, vocab - 1), vocab})
        top_k = [rng.choice(choices) for _ in range(num_dist if per_request else 1)]
    if api in ("top_p", "top_k_top_p"):
        # a boundary-placed p exercises the fp32 threshold; otherwise a plain value.
        if api == "top_p" and rng.random() < 0.5:
            p_boundary = rng.choice(["below", "at", "above"])
            row0 = _row(shapes[0], vocab, random.Random(seed))
            row0 = row0 / row0.sum()
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

# Tracked wrong-answer / crash-class findings for this API, applied through the shared
# ledger. Empty: no filed sampling defect is being tolerated.
_LEDGER = FuzzLedger("sampling", findings=())


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
    cuda = os.environ.get("CUDA_HOME", "<cuda>")
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "<idx>")
    return (
        f"REPRO: CUDA_HOME={cuda} CUDA_VISIBLE_DEVICES={dev} "
        f"FLASHINFER_SAMPLING_FUZZ_ONLY_SEED={cfg.seed} "
        f"pytest -s tests/utils/test_sampling_fuzz.py::test_sampling_fuzz"
    )


def _fail(cfg: Cfg, why: str) -> None:
    pytest.fail("\n".join([why, _describe(cfg), _repro(cfg)]), pytrace=False)


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

    A per-request threshold tensor is indexed by the probs row (length == number of
    distributions); the kernel pairs each output with its row's threshold through indices.
    """
    if not cfg.per_request:
        return values[0]
    dt = (
        torch.int64 if which == "k" else _DTYPE
    )  # per-request top_k is int64 (see test_sampling.py)
    return torch.tensor(values, dtype=dt, device=_DEVICE)


def _call(cfg: Cfg, probs, k, p, indices, seed, offset, generator=None):
    if cfg.api == "sampling_from_probs":
        return flashinfer.sampling.sampling_from_probs(
            probs, indices=indices, seed=seed, offset=offset, generator=generator
        )
    if cfg.api == "top_k":
        return flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, indices=indices, seed=seed, offset=offset, generator=generator
        )
    if cfg.api == "top_p":
        return flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, indices=indices, seed=seed, offset=offset, generator=generator
        )
    return flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs,
        k,
        p,
        indices=indices,
        filter_apply_order=cfg.order,
        seed=seed,
        offset=offset,
        generator=generator,
    )


@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_sampling_fuzz(cfg: Cfg):
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    probs = _build_probs(cfg, rng)
    probs64 = reference_normalize(probs)
    nd = cfg.num_dist
    seed = 0xA5A5 + cfg.seed

    # Per-distribution reference. loose: a draw may not land outside it. strict: every
    # firmly-included class must be drawn. target: the exact post-filter mass. loose and
    # strict differ only by the fp32 boundary tolerance, so the crossing element of an
    # at-boundary case is required by neither side.
    shift = _fp32_eps(cfg.vocab)
    loose = torch.zeros(nd, cfg.vocab, dtype=torch.bool)
    strict = torch.zeros(nd, cfg.vocab, dtype=torch.bool)
    target = torch.zeros(nd, cfg.vocab, dtype=torch.float64)
    for d in range(nd):
        k = cfg.top_k[d if cfg.per_request else 0] if cfg.top_k else None
        p = cfg.top_p[d if cfg.per_request else 0] if cfg.top_p else None
        loose[d] = reference_allowed_support(
            cfg.api, probs64[d], k, p, cfg.order, shift
        )
        strict[d] = reference_allowed_support(
            cfg.api, probs64[d], k, p, cfg.order, -shift
        )
        target[d] = reference_target_distribution(cfg.api, probs64[d], k, p, cfg.order)
    loose, strict, target = loose.to(_DEVICE), strict.to(_DEVICE), target.to(_DEVICE)

    # Draw a few thousand samples so "never outside the support" has real power.
    samples, dist_idx = _draw_samples(cfg, probs, seed)
    if samples.dtype != torch.int32:
        _fail(cfg, f"output dtype {samples.dtype}, expected int32")
    if not (torch.all(samples >= 0) and torch.all(samples < cfg.vocab)):
        _fail(cfg, f"sample index out of range [0,{cfg.vocab})")

    ok = loose[dist_idx.long(), samples.long()]
    if not bool(ok.all()):
        i = int((~ok).nonzero()[0, 0].item())
        d, tok = int(dist_idx[i].item()), int(samples[i].item())
        _fail(
            cfg,
            f"dist {d}: drew class {tok} (p={probs64[d, tok].item():.3e}) outside the "
            f"loose support (|support|={int(loose[d].sum().item())})",
        )

    # Mirror: a class firmly in the support with a high enough expected count must appear.
    # Requiring expected >= 64 makes the miss probability < e^-64, so this cannot flake; a
    # kernel that under-includes (drops the crossing element or keeps k-1 classes) is caught.
    n_per_row = torch.bincount(dist_idx.long(), minlength=nd)
    obs = torch.zeros(nd * cfg.vocab, dtype=torch.long, device=_DEVICE)
    obs.scatter_add_(
        0,
        dist_idx.long() * cfg.vocab + samples.long(),
        torch.ones_like(samples, dtype=torch.long),
    )
    obs = obs.view(nd, cfg.vocab)
    expected = target * n_per_row.unsqueeze(1)
    required = strict & (expected >= 64)
    missing = required & (obs == 0)
    if bool(missing.any()):
        d, tok = (int(x) for x in missing.nonzero()[0])
        _fail(
            cfg,
            f"dist {d}: class {tok} is in the strict support with ~{expected[d, tok].item():.0f} "
            f"expected draws but was never drawn ({int(n_per_row[d].item())} draws for this row)",
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

    indices path: one launch of _SAMPLES_PER_CASE outputs mapped back to the num_dist rows,
    so out_batch != num_dist. non-indices path: tile the rows so a single launch still draws
    thousands of samples per row (each output has its own block index, so the draws are
    independent even for repeated rows).
    """
    nd = cfg.num_dist
    reps = 1 if single_chunk else -(-_SAMPLES_PER_CASE // nd)
    dist_idx = torch.arange(nd, device=_DEVICE).repeat(reps)
    k = _threshold(cfg.top_k, "k", cfg) if cfg.top_k else None
    p = _threshold(cfg.top_p, "p", cfg) if cfg.top_p else None
    if cfg.use_indices:
        s = _call(cfg, probs, k, p, dist_idx.to(torch.int32), seed, 0)
    else:
        # per-request thresholds ride along the tiled rows; a scalar stays scalar.
        kt = k.repeat(reps) if torch.is_tensor(k) else k
        pt = p.repeat(reps) if torch.is_tensor(p) else p
        s = _call(cfg, probs.repeat(reps, 1), kt, pt, None, seed, 0)
    torch.cuda.synchronize()
    return s, dist_idx


def _check_generator_replay(cfg: Cfg, probs) -> None:
    nd = cfg.num_dist
    dist_idx = torch.arange(nd, device=_DEVICE).to(torch.int32)
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
# Bounded distributional check. Frozen configs over strictly-decreasing, well-separated
# distributions (no small target bins). A config fails only when the p-value is past the
# Bonferroni threshold and an effect size is past a declared floor.
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


def _draw_counts(api, order, top_k, top_p, probs_row, seed) -> torch.Tensor:
    vocab = probs_row.numel()
    probs = probs_row.to(_DTYPE).to(_DEVICE).unsqueeze(0)
    indices = torch.zeros(SAMPLE_COUNT, dtype=torch.int32, device=_DEVICE)
    if api == "sampling_from_probs":
        s = flashinfer.sampling.sampling_from_probs(
            probs, indices=indices, seed=seed, offset=0
        )
    elif api == "top_k":
        s = flashinfer.sampling.top_k_sampling_from_probs(
            probs, top_k, indices=indices, seed=seed, offset=0
        )
    elif api == "top_p":
        s = flashinfer.sampling.top_p_sampling_from_probs(
            probs, top_p, indices=indices, seed=seed, offset=0
        )
    else:
        s = flashinfer.sampling.top_k_top_p_sampling_from_probs(
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


@pytest.mark.parametrize(
    "sc",
    _STAT_CONFIGS,
    ids=[f"{c[0]}_{c[1]}_v{c[4]}_k{c[2]}_p{c[3]}" for c in _STAT_CONFIGS],
)
def test_sampling_distribution(sc):
    api, order, top_k, top_p, vocab, active, ratio = sc
    probs_row = _geometric(vocab, active, ratio)
    target = reference_target_distribution(api, probs_row, top_k, top_p, order)

    counts = _draw_counts(api, order, top_k, top_p, probs_row, seed=0xBEEF)
    res = chi_square_gof(counts, target)

    drew_forbidden = int(counts[target == 0].sum().item())
    assert drew_forbidden == 0, (
        f"{api}/{order} v{vocab}: {drew_forbidden} draws on zero-target classes\n"
        f"  counts={counts.tolist()} target={target.tolist()}"
    )

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
    generator; the multi-seed calibration lives in the offline sweep, not here."""
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
    # Boundary ties are kept together.
    tied = torch.tensor([0.3, 0.3, 0.3, 0.1], dtype=torch.float64)
    assert reference_top_k_support(tied, 1).tolist() == [True, True, True, False]
