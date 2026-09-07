"""Contract values shared by the controller, selection, and backends.

Mirrors ``flashinfer/mla/_batch_mla/_contracts.py``: the immutable values that
fix what a plan means (``PlanMetadata``), what a resolution promised
(``Resolution``), and the loud-error helpers every layer uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from ._backends._capabilities import MIN_DENSE_PAGE_SIZE


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _expect_window_left(window_left: int) -> None:
    # trtllm's launcher special-cases exactly -1 (unlimited); other negatives
    # become a force-enabled negative window there while fa/cudnn read them
    # as unlimited — backend-divergent, so reject anything below -1 loudly.
    _expect(
        isinstance(window_left, int) and window_left >= -1,
        f"window_left must be >= -1 (-1 = unlimited), got {window_left!r}",
    )


def _expect_page_size(page_size: int, kv_input_form: str) -> None:
    _expect(
        isinstance(page_size, int) and page_size >= 1,
        f"page_size must be a positive host int, got {page_size!r}",
    )
    if kv_input_form == "block_tables":
        _expect(
            page_size >= MIN_DENSE_PAGE_SIZE,
            f"page_size {page_size} < {MIN_DENSE_PAGE_SIZE} with the dense "
            "block_tables form: a dense table at token-granular page sizes "
            "degenerates to (batch, max_context) — pass the flat "
            "kv_page_indices form instead (any page_size >= 1)",
        )


def resolve_config_key(
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    q_dtype,
    page_size,
    kv_layout,
    causal,
    need_lse,
    window_left,
    kv_input_form,
) -> Tuple:
    """The observational config a Resolution is pinned to (drift detection)."""
    return (
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        str(q_dtype),
        page_size,
        kv_layout,
        causal,
        need_lse,
        window_left,
        kv_input_form,
    )


@dataclass(frozen=True)
class Resolution:
    """Init-time resolution result (proposal §5.3, level 1).

    ``backends`` is the pinned, ordered candidate set: every member is
    runnable for the declared configuration and observationally identical at
    the contract level (same dtypes, same LSE availability), so a later
    plan-time choice within this set cannot surprise the engine.  Pass the
    whole Resolution to ``plan(backend=...)`` to enforce the pinning:
    plan() verifies its arguments match ``config`` and chooses only within
    ``backends``.
    """

    backends: Tuple[str, ...]
    excluded: Dict[str, str] = field(default_factory=dict)
    kv_layout: str = "HND"
    # the resolve-time observational config, used by plan() to detect drift
    config: Tuple = ()

    @property
    def chosen(self) -> str:
        return self.backends[0]

    def explain(self) -> str:
        lines = [f"candidates (preference order): {list(self.backends)}"]
        for name, reason in self.excluded.items():
            lines.append(f"excluded {name}: {reason}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PlanMetadata:
    """Everything a backend may read about one planned batch.

    Built by the controller after validation; backends receive it together
    with the derived forms (``_planning.Derived``) and must not reach past it
    into caller state. ``block_tables`` is the dense table — given by the
    caller or derived — and is ``None`` only when the chosen backend does not
    need it.
    """

    qo_indptr: torch.Tensor
    kv_seq_lens: torch.Tensor
    block_tables: Optional[torch.Tensor]
    kv_input_form: str
    page_size: int
    max_q_len: int
    max_kv_len: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    causal: bool
    window_left: int
    kv_layout: str
    sm_scale: float
    return_lse: bool
    batch_size: int
    qo_indptr_cpu: torch.Tensor
    kv_seq_lens_cpu: torch.Tensor

    @property
    def total_q_tokens(self) -> int:
        return int(self.qo_indptr_cpu[-1])


__all__ = [
    "PlanMetadata",
    "Resolution",
    "resolve_config_key",
]
