"""Contract values shared by the controller, selection, and backends.

Mirrors ``flashinfer/mla/_batch_mla/_contracts.py``: the immutable values that
fix what a plan means (``PlanMetadata``), what a resolution promised
(``Resolution``), and the loud-error helpers every layer uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from ._backends._capabilities import MIN_DENSE_PAGE_SIZE

LSE_MODES = ("none", "base2", "basee")
LN2 = math.log(2.0)


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _expect_lse_mode(lse_mode: str) -> None:
    _expect(
        lse_mode in LSE_MODES,
        f"lse_mode must be one of {LSE_MODES} (MLA vocabulary: none / base-2 / "
        f"natural log), got {lse_mode!r}",
    )


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
    kv_dtype,
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
        str(kv_dtype),
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


@dataclass(frozen=True, eq=False)
class PagedAttentionMetadata:
    """The canonical per-batch metadata for paged attention (one object).

    Build it once per scheduler step with :meth:`dense` (vLLM-style block
    table) or :meth:`csr` (sglang-style flat page ids); the two constructors
    make "exactly one paging form" structural.  Construction validates shapes
    and values against the host mirrors — pass the mirrors the engine already
    owns and construction is zero-sync; otherwise it performs ONE documented
    D2H here, and every later ``plan()`` on this object is sync-free.

    The object also owns the backend-neutral derived forms (CSR page indptr,
    cumulative KV lengths, dense table) lazily, so several plans over the same
    batch (windowed / full layers, causal / non-causal) derive once.

    Identity semantics: two objects compare by identity, not by tensor
    contents (they are meant to be built once per step and reused).
    """

    qo_indptr: torch.Tensor  # (b+1,) int32 device, token-unit prefix sums
    kv_seq_lens: torch.Tensor  # (b,) int32 device, per-request valid KV lengths
    page_size: int
    max_q_len: int
    max_kv_len: int
    block_tables: Optional[torch.Tensor] = None  # (b, max_pages) dense form
    kv_page_indices: Optional[torch.Tensor] = None  # flat CSR page ids
    qo_indptr_cpu: Optional[torch.Tensor] = None
    kv_seq_lens_cpu: Optional[torch.Tensor] = None
    _derived: Dict[bool, Any] = field(default_factory=dict, repr=False)

    # ---- constructors ----
    @classmethod
    def dense(
        cls,
        qo_indptr: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        *,
        page_size: int,
        max_q_len: int,
        max_kv_len: int,
        qo_indptr_cpu: Optional[torch.Tensor] = None,
        kv_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> "PagedAttentionMetadata":
        """vLLM-style: a dense ``(batch, max_pages_per_seq)`` block table (page_size >= 8)."""
        return cls(
            qo_indptr=qo_indptr,
            kv_seq_lens=kv_seq_lens,
            page_size=page_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            block_tables=block_tables,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_seq_lens_cpu,
        )

    @classmethod
    def csr(
        cls,
        qo_indptr: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        kv_page_indices: torch.Tensor,
        *,
        page_size: int,
        max_q_len: int,
        max_kv_len: int,
        qo_indptr_cpu: Optional[torch.Tensor] = None,
        kv_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> "PagedAttentionMetadata":
        """sglang-style: flat CSR page ids in request order (any page_size >= 1).

        Page-unit indptr and last-page lengths are NOT accepted: they derive
        from ``kv_seq_lens`` + ``page_size``; a second copy would be a second
        truth.
        """
        return cls(
            qo_indptr=qo_indptr,
            kv_seq_lens=kv_seq_lens,
            page_size=page_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            kv_page_indices=kv_page_indices,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_seq_lens_cpu,
        )

    def __post_init__(self):
        from ._planning import validate_structure, validate_values

        _expect(
            (self.block_tables is None) != (self.kv_page_indices is None),
            "pass EXACTLY ONE paging form: PagedAttentionMetadata.dense(block_tables) "
            "or .csr(kv_page_indices)",
        )
        _expect(
            isinstance(self.kv_seq_lens, torch.Tensor) and self.kv_seq_lens.is_cuda,
            "kv_seq_lens must be a CUDA tensor",
        )
        validate_structure(
            self.kv_seq_lens.device,
            self.qo_indptr,
            self.kv_seq_lens,
            self.block_tables,
            self.kv_page_indices,
            self.page_size,
            self.max_q_len,
            self.max_kv_len,
            self.kv_input_form,
        )
        # Value-level validation is unconditional — it is what makes the
        # reject-or-correct property hold.  Zero-sync iff the caller hands us
        # the host mirrors it already owns; otherwise ONE documented D2H here.
        if self.qo_indptr_cpu is None:
            object.__setattr__(self, "qo_indptr_cpu", self.qo_indptr.cpu())
        if self.kv_seq_lens_cpu is None:
            object.__setattr__(self, "kv_seq_lens_cpu", self.kv_seq_lens.cpu())
        validate_values(
            self.qo_indptr,
            self.kv_seq_lens,
            self.block_tables,
            self.kv_page_indices,
            self.page_size,
            self.max_q_len,
            self.max_kv_len,
            self.qo_indptr_cpu,
            self.kv_seq_lens_cpu,
            causal=False,  # the causal envelope depends on plan(causal=...)
        )

    # ---- derived facts ----
    @property
    def kv_input_form(self) -> str:
        return "block_tables" if self.block_tables is not None else "page_indices"

    @property
    def device(self) -> torch.device:
        return self.kv_seq_lens.device

    @property
    def batch_size(self) -> int:
        return int(self.kv_seq_lens.shape[0])

    @property
    def total_q_tokens(self) -> int:
        assert self.qo_indptr_cpu is not None
        return int(self.qo_indptr_cpu[-1])

    def derived(self, *, needs_dense: bool):
        """Backend-neutral derived forms, computed once per (object, needs_dense)."""
        from ._planning import derive

        key = bool(needs_dense) or self.block_tables is not None
        d = self._derived.get(key)
        if d is None:
            d = derive(
                self.qo_indptr,
                self.kv_seq_lens,
                self.block_tables,
                self.kv_page_indices,
                self.page_size,
                self.max_kv_len,
                needs_dense=key,
            )
            self._derived[key] = d
        return d


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
    lse_mode: str  # "none" | "base2" | "basee"
    batch_size: int
    qo_indptr_cpu: torch.Tensor
    kv_seq_lens_cpu: torch.Tensor

    @property
    def total_q_tokens(self) -> int:
        return int(self.qo_indptr_cpu[-1])

    @property
    def need_lse(self) -> bool:
        return self.lse_mode != "none"


__all__ = [
    "LN2",
    "LSE_MODES",
    "PagedAttentionMetadata",
    "PlanMetadata",
    "Resolution",
    "resolve_config_key",
]
