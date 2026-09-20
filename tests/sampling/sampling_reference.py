"""Independent CPU oracle for ``flashinfer.sampling``.

Written from the kernels at the pinned revision (``include/flashinfer/sampling.cuh``,
``include/flashinfer/topk.cuh``, ``include/flashinfer/air_top_p.cuh``) and deliberately shares
no code with ``flashinfer``: it re-implements softmax and every filter itself, so a wrong filter
in the library cannot be mirrored by the oracle.  It runs on CPU-only tensors in float64 and
never launches a kernel.

The filter semantics it encodes (NOTES.md records the source line each fact came from):

* Retention is *value based*, so ties at a filter boundary are kept together -- unlike
  ``torch.topk``, whose tie order is not a documented contract:
    - top-k:  ``count{p_j > p_i} < k``
    - top-p:  ``sum{p_j > p_i} < top_p`` (strict tail comparison, so a token whose strictly
      larger tail mass is exactly ``top_p`` is NOT retained: the boundary token enters the
      nucleus once the cumulative mass has reached ``top_p``)
    - min-p:  ``p_i >= max(p) * min_p`` (inclusive, so a token exactly at the relative
      threshold IS retained -- the opposite boundary convention from top-p)
* ``top_k_first`` and ``joint`` are different filters and get different references:
    - ``top_k_first`` = the top-p nucleus of the distribution renormalized over the top-k set,
    - ``joint``       = the intersection of the top-k set and the top-p nucleus.
  Neither can stand in for the other.
* The distribution over the retained set is that set's probabilities renormalized (the kernels'
  rejection search is exact for this target), and zero-probability tokens never receive mass.

Every row of ``probs`` must be non-negative; the sampling APIs assume each row is normalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

ORDER_TOP_K_FIRST = "top_k_first"
ORDER_JOINT = "joint"
# Per-row filter parameters are keyed by the probability row they describe; KEYING_BLOCK
# reproduces the original block-index keying so a test can prove its case detects that defect.
KEYING_ROW = "row"
KEYING_BLOCK = "block"


def _f64(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(
            f"expected a 2D (rows, vocab) tensor, got shape {tuple(x.shape)}"
        )
    t = x if x.dtype == torch.float64 else x.double()
    if t.numel() and (bool(torch.isnan(t).any()) or float(t.min()) < 0.0):
        raise ValueError("the oracle needs non-negative probabilities")
    return t


def reference_softmax(
    logits: torch.Tensor, temperature=None, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """fp64 softmax with an optional scalar or per-row temperature."""
    z = logits.to(dtype)
    if temperature is not None:
        t = torch.as_tensor(temperature, dtype=dtype)
        z = z / (t.unsqueeze(-1) if t.dim() == 1 else t)
    z = z - z.max(dim=-1, keepdim=True).values
    e = torch.exp(z)
    return e / e.sum(dim=-1, keepdim=True)


def group_stats(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per entry: (mass of strictly larger entries, count of strictly larger entries).

    Computed in fp64 from a descending sort.  Both quantities are constant inside a group of
    equal probabilities, which is what makes the retention rules value based.
    """
    p = _f64(probs)
    rows, vocab = p.shape
    sorted_p, order = torch.sort(p, dim=-1, descending=True)
    excl = torch.cumsum(sorted_p, dim=-1) - sorted_p
    idx = torch.arange(vocab).expand(rows, vocab)
    is_new = torch.ones_like(sorted_p, dtype=torch.bool)
    is_new[..., 1:] = sorted_p[..., 1:] != sorted_p[..., :-1]
    # cummax over group starts: excl is non-decreasing on a descending sort, so the largest
    # group-start value seen so far is this group's own value, and the last group start seen is
    # this group's first position.
    mass_at_start = torch.cummax(
        torch.where(is_new, excl, torch.zeros_like(excl)), dim=-1
    ).values
    pos_at_start = torch.cummax(
        torch.where(is_new, idx, torch.full_like(idx, -1)), dim=-1
    ).values
    mass = torch.empty_like(sorted_p).scatter_(-1, order, mass_at_start)
    count = torch.empty_like(sorted_p, dtype=torch.int64).scatter_(
        -1, order, pos_at_start
    )
    return mass, count


def source_rows(
    n_prob_rows: int, indices: Optional[torch.Tensor], batch_size: Optional[int] = None
) -> torch.Tensor:
    """Output row -> probability row, i.e. the ``indices`` contract (identity when None)."""
    if indices is None:
        rows = n_prob_rows if batch_size is None else batch_size
        return torch.arange(rows, dtype=torch.int64)
    idx = torch.as_tensor(indices, dtype=torch.int64).reshape(-1)
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n_prob_rows):
        raise ValueError(
            f"indices out of range: values in [{int(idx.min())}, {int(idx.max())}] for "
            f"{n_prob_rows} probability rows"
        )
    return idx


def expand_param(
    param,
    *,
    n_prob_rows: int,
    indices: Optional[torch.Tensor],
    keying: str,
) -> Optional[torch.Tensor]:
    """Per output row parameter values, under the declared keying rule.

    The library validates a per-row parameter tensor against ``probs.size(0)`` (the number of
    probability rows) even when ``indices`` reuses those rows for more outputs, so the same
    length check is applied here.
    """
    if param is None:
        return None
    t = torch.as_tensor(param, dtype=torch.float64)
    src = source_rows(n_prob_rows, indices)
    if t.dim() == 0:
        return t.expand(src.numel())
    if t.dim() != 1:
        raise ValueError(
            f"filter parameter must be 0D or 1D, got shape {tuple(t.shape)}"
        )
    if t.numel() != n_prob_rows:
        raise ValueError(
            f"filter parameter length {t.numel()} does not match the {n_prob_rows} "
            "probability rows"
        )
    if keying == KEYING_ROW:
        return t[src]
    if keying == KEYING_BLOCK:
        # The original keying: parameter index == output row index.  A parameter tensor shorter
        # than the batch is read past its end by the kernel (the defect this keying models), so
        # wrap it to keep the reference total.
        return t[torch.arange(src.numel()) % t.numel()]
    raise ValueError(f"unknown keying {keying!r}")


@dataclass(frozen=True)
class FilterSpec:
    """One filter configuration, applied to the rows of a probability matrix."""

    top_k: object = None
    top_p: object = None
    min_p: object = None
    order: str = ORDER_TOP_K_FIRST

    def label(self) -> str:
        parts = []
        if self.top_k is not None:
            parts.append(f"k={self.top_k}")
        if self.top_p is not None:
            parts.append(f"p={self.top_p}")
        if self.min_p is not None:
            parts.append(f"min_p={self.min_p}")
        parts.append(self.order)
        return ",".join(str(x) for x in parts)


def retained_mask(
    probs: torch.Tensor,
    spec: FilterSpec,
    *,
    indices: Optional[torch.Tensor] = None,
    keying: str = KEYING_ROW,
) -> torch.Tensor:
    """Boolean (output rows, vocab) mask of the tokens a filter combination retains."""
    p = _f64(probs)
    n_prob_rows = p.shape[0]
    rows = p.index_select(0, source_rows(n_prob_rows, indices))
    mass, count = group_stats(rows)
    keep = torch.ones_like(rows, dtype=torch.bool)
    top_k = expand_param(
        spec.top_k, n_prob_rows=n_prob_rows, indices=indices, keying=keying
    )
    top_p = expand_param(
        spec.top_p, n_prob_rows=n_prob_rows, indices=indices, keying=keying
    )
    if spec.order not in (ORDER_TOP_K_FIRST, ORDER_JOINT):
        raise ValueError(f"unknown filter order {spec.order!r}")
    if top_k is not None:
        keep &= count < top_k.unsqueeze(-1)
    if top_p is not None:
        if spec.order == ORDER_JOINT or top_k is None:
            keep &= mass < top_p.unsqueeze(-1)
        else:
            # top_k_first: renormalize over the top-k set and take the nucleus of THAT
            # distribution, so the tail mass is measured after renormalization.
            renorm = torch.where(keep, rows, torch.zeros_like(rows))
            z = renorm.sum(dim=-1, keepdim=True)
            q = renorm / z.clamp_min(torch.finfo(torch.float64).tiny)
            q_mass, _ = group_stats(q)
            keep &= q_mass < top_p.unsqueeze(-1)
    if spec.min_p is not None:
        min_p = expand_param(
            spec.min_p, n_prob_rows=n_prob_rows, indices=indices, keying=keying
        )
        keep &= rows >= rows.max(dim=-1, keepdim=True).values * min_p.unsqueeze(-1)
    return keep


def target_distribution(
    probs: torch.Tensor,
    spec: Optional[FilterSpec] = None,
    *,
    indices: Optional[torch.Tensor] = None,
    keying: str = KEYING_ROW,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(support, target) for the sampling APIs.

    ``support`` is the retained, positive-probability mask; ``target`` is ``probs`` renormalized
    over that support, with probabilities the kernel cannot reach (zeros) pinned to zero.  A row
    with an empty support yields an all-zero target row: no legal token exists for it, which the
    sampling APIs report through ``valid`` (see :func:`degenerate_expectation`).
    """
    if spec is None:
        spec = FilterSpec()
    p = _f64(probs)
    rows = p.index_select(0, source_rows(p.shape[0], indices))
    keep = retained_mask(p, spec, indices=indices, keying=keying) & (rows > 0)
    z = torch.where(keep, rows, torch.zeros_like(rows)).sum(dim=-1, keepdim=True)
    target = torch.where(keep, rows, torch.zeros_like(rows)) / torch.where(
        z > 0, z, torch.ones_like(z)
    )
    return keep, target


def degenerate_expectation(
    probs_row: torch.Tensor, api: str, spec: FilterSpec
) -> Tuple[bool, int]:
    """What a sampling API returns for a row with no legal token: ``(valid, sample)``.

    Every sampling API writes ``0`` with ``valid=False`` -- except ``min_p_sampling_from_probs``
    on a row with no positive probability: its predicate ``p >= max(p) * min_p`` is true for an
    all-zero row (``pivot == 0``), so the scan records a last valid index and the kernel's
    never-crossed fallback returns ``vocab_size - 1`` with ``valid=True``.
    """
    if api == "min_p_sampling_from_probs" and not bool((probs_row > 0).any()):
        return True, int(probs_row.numel() - 1)
    return False, 0


def inverse_cdf_sample(
    target: torch.Tensor, draws: int, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Draw ``draws`` classes per row from ``target`` by explicit inverse CDF (CPU reference)."""
    rows, vocab = target.shape
    cdf = torch.cumsum(target, dim=-1).contiguous()
    u = torch.rand((rows, draws), generator=generator, dtype=target.dtype)
    # First index whose cumulative mass strictly exceeds u: a zero-mass class can never be the
    # answer, so zero-probability tokens are unreachable by construction.
    return torch.searchsorted(cdf, u.contiguous(), right=True).clamp_(max=vocab - 1)


def frequencies(samples: torch.Tensor, vocab: int) -> torch.Tensor:
    """Row-wise observed frequencies of ``samples`` over ``vocab`` classes."""
    counts = torch.zeros((samples.shape[0], vocab), dtype=torch.float64)
    counts.scatter_add_(
        1, samples.to(torch.int64), torch.ones_like(samples, dtype=torch.float64)
    )
    return counts / samples.shape[1]


def effective_classes(target: torch.Tensor) -> int:
    """Classes a case can actually draw: its declared comparison count."""
    return int((target > 0).any(dim=0).sum())
