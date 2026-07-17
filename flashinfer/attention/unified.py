"""Unified paged-prefill — PROTOTYPE for API review.

This module is the working prototype of ``PAGED_PREFILL_UNIFICATION_PROPOSAL.md``.
It is deliberately self-contained so reviewers can read one file top-to-bottom;
the production landing is a merge into ``BatchPrefillWithPagedKVCacheWrapper``
(everything here dispatches to *existing* kernels — there is no new kernel).

Three layers (see proposal §"Architecture"):

    contract   UnifiedPagedPrefill.plan()/run()  +  resolve_paged_prefill()
    adapter    _FaAdapter / _CudnnAdapter / _TrtllmGenAdapter   (thin, private)
    kernel     the existing wrapper / standalone functions      (untouched)

Design rules enforced here (each traces to a documented failure mode):

1.  ONE canonical metadata form — token-unit ``qo_indptr``, per-request
    ``kv_seq_lens``, a dense ``block_tables``, and REQUIRED host maxes.
    Everything any backend wants (CSR page indices for fa2/fa3, cumulative KV
    lens for trtllm-gen, ``(b,1,1,1)`` lens for cuDNN) is derived internally.
2.  Closed input set + loud errors.  Combinations outside the contract raise
    ``ValueError`` with the fix in the message.  We never guess a layout
    (cuDNN issue #3800 is what guessing looks like).
3.  Reject-or-correct.  Anything this API returns must match the reference
    semantics; anything it cannot address must raise.  The companion fuzzer
    (``tests/attention/test_unified_prefill_fuzzer.py``) enforces exactly this
    property with randomized valid and corrupted inputs.
4.  Two-level selection.  ``resolve_paged_prefill()`` is a static, tensor-free
    query usable at engine init (before pool allocation / graph capture); the
    plan-time choice may only pick within the resolved candidate set.
5.  One output contract.  LSE is always base-2, shape ``(total_q_tokens,
    num_qo_heads)``, fp32 — adapters normalize native formats (cuDNN returns
    natural-log padded ``(b, max_q, h)`` stats; the fold lives in its adapter,
    not in callers).

Prototype simplifications (documented, not hidden):
- ``kv_layout="HND"`` only (pages, H, page_size, D).  NHD is a capability
  axis in the proposal; wiring it is mechanical.
- dtypes: fp16/bf16 only.  fp8/nvfp4 are capability axes, out of scope here.
- Heuristic order is a static per-arch placeholder, to be seeded from the
  benchmark suite (proposal §5.2).  Autotune hook (§5.4) is not wired.
- Each adapter allocates its own workspace lazily (production: share).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

__all__ = [
    "BackendCapability",
    "Resolution",
    "resolve_paged_prefill",
    "UnifiedPagedPrefill",
]

_LOG2E = math.log2(math.e)


# --------------------------------------------------------------------------
# Capability layer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendCapability:
    """Static description of what one backend can run.

    This is the queryable capability matrix from proposal §5.1 — the thing
    that replaces consumer-side support tables (vLLM's scattered gates,
    sglang's whitelists).  ``availability_probe`` returns None when the
    backend is usable in this process, else a human-readable reason.
    """

    name: str
    # compute-capability majors (minor-insensitive for the prototype)
    cc_majors: frozenset
    q_dtypes: frozenset
    head_dims: frozenset  # of (head_dim_qk, head_dim_vo) pairs
    page_sizes: Optional[frozenset]  # None = any
    kv_layouts: frozenset
    supports_lse: bool
    supports_noncausal: bool
    requires_contiguous_q: bool
    # native LSE format, normalized by the adapter:
    #   "base2_tokens_h"  — already the contract
    #   "ln_padded_bsh"   — natural-log (b, max_q, h); adapter gathers + folds
    lse_native: str = "base2_tokens_h"

    def check(
        self,
        *,
        cc_major: int,
        q_dtype: torch.dtype,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        kv_layout: str,
        causal: bool,
        need_lse: bool,
    ) -> Optional[str]:
        """Return None if runnable, else the exclusion reason (for explain())."""
        if cc_major not in self.cc_majors:
            return f"unsupported compute capability sm_{cc_major}x"
        if q_dtype not in self.q_dtypes:
            return f"unsupported q dtype {q_dtype}"
        if (head_dim_qk, head_dim_vo) not in self.head_dims:
            return f"unsupported head dims ({head_dim_qk}, {head_dim_vo})"
        if self.page_sizes is not None and page_size not in self.page_sizes:
            return f"unsupported page_size {page_size} (supported: {sorted(self.page_sizes)})"
        if kv_layout not in self.kv_layouts:
            return f"unsupported kv_layout {kv_layout}"
        if not causal and not self.supports_noncausal:
            return "non-causal attention not supported"
        if need_lse and not self.supports_lse:
            return "LSE output not supported"
        return None


def _probe_fa(backend: str) -> Optional[str]:
    if backend == "fa3":
        from ..utils import is_sm90a_supported

        if not is_sm90a_supported(torch.device("cuda")):
            return "fa3 requires SM90a (Hopper) and CUDA >= 12.3"
    return None


def _probe_cudnn() -> Optional[str]:
    from ..cudnn import prefill as cudnn_prefill

    if not cudnn_prefill.CUDNN_AVAILABLE:
        return "cudnn-frontend python package not importable"
    return None


def _probe_trtllm() -> Optional[str]:
    # Cubin availability is a real capability question (proposal: it should be
    # a library answer, not an engine-side HTTP probe).  The prototype defers
    # to first-run download; a production probe would consult the local cubin
    # cache / FLASHINFER_NO_DOWNLOAD.
    return None


_F16 = frozenset({torch.float16, torch.bfloat16})
_D128 = frozenset({(128, 128)})

CAPABILITIES: Dict[str, BackendCapability] = {
    "fa2": BackendCapability(
        name="fa2",
        cc_majors=frozenset({8, 9, 10, 12}),
        q_dtypes=_F16,
        head_dims=frozenset({(64, 64), (128, 128), (256, 256)}),
        page_sizes=None,
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        supports_noncausal=True,
        requires_contiguous_q=False,
    ),
    "fa3": BackendCapability(
        name="fa3",
        cc_majors=frozenset({9}),
        q_dtypes=_F16,
        head_dims=frozenset({(64, 64), (128, 128), (256, 256), (192, 128)}),
        page_sizes=None,
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        supports_noncausal=True,
        requires_contiguous_q=False,
    ),
    "cudnn": BackendCapability(
        name="cudnn",
        cc_majors=frozenset({8, 9, 10, 12}),
        q_dtypes=_F16,
        head_dims=frozenset({(128, 128), (192, 128)}),
        page_sizes=None,
        kv_layouts=frozenset({"HND"}),
        supports_lse=True,
        supports_noncausal=False,  # cuDNN SDPA graph path here is causal-only
        requires_contiguous_q=True,  # token-unit offsets assume packed THD
        lse_native="ln_padded_bsh",
    ),
    "trtllm-gen": BackendCapability(
        name="trtllm-gen",
        cc_majors=frozenset({10}),
        q_dtypes=_F16,
        head_dims=_D128,
        page_sizes=frozenset({16, 32, 64}),
        kv_layouts=frozenset({"HND"}),
        supports_lse=True,
        supports_noncausal=False,
        requires_contiguous_q=True,  # TMA: last dim stride 1; keep strict here
    ),
}

_PROBES = {
    "fa2": lambda: _probe_fa("fa2"),
    "fa3": lambda: _probe_fa("fa3"),
    "cudnn": _probe_cudnn,
    "trtllm-gen": _probe_trtllm,
}

# Static heuristic placeholder (proposal §5.2: to be seeded from the benchmark
# suite; it only has to beat consumer tables that rot).  Order = preference.
_HEURISTIC_ORDER = {
    10: ("trtllm-gen", "cudnn", "fa2"),
    9: ("fa3", "fa2", "cudnn"),
    8: ("fa2", "cudnn"),
    12: ("fa2", "cudnn"),
}


@dataclass(frozen=True)
class Resolution:
    """Init-time resolution result (proposal §5.3, level 1).

    ``backends`` is the pinned, ordered candidate set: every member is
    runnable for the declared configuration and observationally identical at
    the contract level (same dtypes, same LSE availability), so a later
    plan-time choice within this set cannot surprise the engine.
    """

    backends: Tuple[str, ...]
    excluded: Dict[str, str] = field(default_factory=dict)
    kv_layout: str = "HND"

    @property
    def chosen(self) -> str:
        return self.backends[0]

    def explain(self) -> str:
        lines = [f"candidates (preference order): {list(self.backends)}"]
        for name, reason in self.excluded.items():
            lines.append(f"excluded {name}: {reason}")
        return "\n".join(lines)


def resolve_paged_prefill(
    *,
    device: Optional[torch.device] = None,
    cc_major: Optional[int] = None,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: Optional[int] = None,
    q_dtype: torch.dtype,
    page_size: int,
    kv_layout: str = "HND",
    causal: bool = True,
    need_lse: bool = False,
    backend: str = "auto",
) -> Resolution:
    """Static backend resolution — no wrapper, no tensors.

    Callable at engine init, before the KV pool is allocated and before any
    CUDA graph is captured (vLLM decides its cudagraph mode and Q dtype at
    that point).  Raises ``ValueError`` when nothing can run, with per-backend
    reasons — the "explain" answer that consumer-side tables cannot give.
    """
    if head_dim_vo is None:
        head_dim_vo = head_dim_qk
    if cc_major is None:
        dev = device if device is not None else torch.device("cuda")
        cc_major = torch.cuda.get_device_properties(dev).major
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads}) for GQA/MQA"
        )

    order = _HEURISTIC_ORDER.get(cc_major, ())
    if backend != "auto":
        if backend not in CAPABILITIES:
            raise ValueError(
                f"unknown backend {backend!r}; known: {sorted(CAPABILITIES)} or 'auto'"
            )
        evaluate: Tuple[str, ...] = (backend,)
    else:
        # Evaluate EVERY known backend so explain() is complete: candidates
        # are ordered by the heuristic; everything else carries its reason.
        evaluate = tuple(order) + tuple(n for n in CAPABILITIES if n not in order)

    candidates, excluded = [], {}
    for name in evaluate:
        cap = CAPABILITIES[name]
        reason = cap.check(
            cc_major=cc_major,
            q_dtype=q_dtype,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=causal,
            need_lse=need_lse,
        )
        if reason is None:
            reason = _PROBES[name]()
        if reason is None:
            candidates.append(name)
        else:
            excluded[name] = reason

    if not candidates:
        detail = "; ".join(f"{k}: {v}" for k, v in excluded.items())
        raise ValueError(f"no runnable backend for this configuration ({detail})")
    return Resolution(
        backends=tuple(candidates), excluded=excluded, kv_layout=kv_layout
    )


# --------------------------------------------------------------------------
# Contract layer
# --------------------------------------------------------------------------


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


class UnifiedPagedPrefill:
    """Unified paged-prefill entry point (prototype).

    Usage (engine-shaped; see ``prototype_demo_unified_prefill.py``)::

        res = resolve_paged_prefill(cc_major=9, num_qo_heads=8, num_kv_heads=2,
                                    head_dim_qk=128, q_dtype=torch.bfloat16,
                                    page_size=16, need_lse=True)
        attn = UnifiedPagedPrefill(device)
        attn.plan(qo_indptr=..., kv_seq_lens=..., block_tables=...,
                  page_size=16, max_q_len=..., max_kv_len=...,
                  num_qo_heads=8, num_kv_heads=2, head_dim_qk=128,
                  q_dtype=torch.bfloat16, causal=True, return_lse=True,
                  backend=res.chosen)
        out, lse = attn.run(q, (k_cache, v_cache))

    Canonical metadata (all CUDA int32, proposal §3.1):

    - ``qo_indptr``:   ``(b+1,)`` token-unit prefix sums of query lengths
    - ``kv_seq_lens``: ``(b,)``  per-request valid KV lengths (masking truth)
    - ``block_tables``: ``(b, max_pages_per_seq)`` dense page table
    - ``page_size, max_q_len, max_kv_len``: host ints (REQUIRED — no hidden
      device→host sync for maxes, ever)
    - optional ``qo_indptr_cpu / kv_seq_lens_cpu`` host mirrors: with them,
      plan() is fully zero-sync (validation + fa2/fa3 scheduling read the
      mirrors — engines already own these on CPU); without them plan()
      performs ONE documented D2H here.  There is no hidden sync and no
      unvalidated path: value-level validation always runs.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = (
            torch.device(device) if device is not None else torch.device("cuda")
        )
        self._adapters: Dict[str, Any] = {}
        self._planned = False

    # ------------------------------ plan ------------------------------

    def plan(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        page_size: int,
        max_q_len: int,
        max_kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: Optional[int] = None,
        q_dtype: torch.dtype,
        kv_dtype: Optional[torch.dtype] = None,
        causal: bool = True,
        sm_scale: Optional[float] = None,
        return_lse: bool = False,
        qo_indptr_cpu: Optional[torch.Tensor] = None,
        kv_seq_lens_cpu: Optional[torch.Tensor] = None,
        backend: str = "auto",
    ) -> "UnifiedPagedPrefill":
        head_dim_vo = head_dim_vo if head_dim_vo is not None else head_dim_qk
        kv_dtype = kv_dtype if kv_dtype is not None else q_dtype

        self._validate_structure(
            qo_indptr, kv_seq_lens, block_tables, page_size, max_q_len, max_kv_len
        )
        # Value-level validation is unconditional — it is what makes the
        # reject-or-correct property hold.  Zero-sync iff the caller hands us
        # the host mirrors it already owns (engines build metadata on CPU);
        # otherwise ONE documented D2H here, never a hidden one later.
        if qo_indptr_cpu is None:
            qo_indptr_cpu = qo_indptr.cpu()
        if kv_seq_lens_cpu is None:
            kv_seq_lens_cpu = kv_seq_lens.cpu()
        self._validate_values(
            qo_indptr,
            kv_seq_lens,
            block_tables,
            page_size,
            max_q_len,
            max_kv_len,
            qo_indptr_cpu,
            kv_seq_lens_cpu,
        )

        resolution = resolve_paged_prefill(
            device=self.device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_dtype=q_dtype,
            page_size=page_size,
            causal=causal,
            need_lse=return_lse,
            backend=backend,
        )
        # Plan-time choice within the resolved set (level 2).  The prototype
        # takes the heuristic head; the autotune hook (proposal §5.4) would
        # consult its cache here, keyed on bucketed (total_q_tokens, max_kv_len).
        self._resolution = resolution
        self._backend = resolution.chosen

        b = kv_seq_lens.shape[0]
        self._meta = dict(
            qo_indptr=qo_indptr,
            kv_seq_lens=kv_seq_lens,
            block_tables=block_tables,
            page_size=page_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            causal=causal,
            sm_scale=sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim_qk),
            return_lse=return_lse,
            batch_size=b,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_seq_lens_cpu,
        )
        self._derived = _derive(qo_indptr, kv_seq_lens, block_tables, page_size)

        adapter = self._adapters.get(self._backend)
        if adapter is None:
            adapter = _ADAPTERS[self._backend](self.device)
            self._adapters[self._backend] = adapter
        adapter.plan(self._meta, self._derived)
        self._adapter = adapter
        self._planned = True
        return self

    def _validate_structure(
        self, qo_indptr, kv_seq_lens, block_tables, page_size, max_q_len, max_kv_len
    ) -> None:
        """Closed-set structural validation. Cheap (host-only, shape-derived)."""
        for name, t, dim in (
            ("qo_indptr", qo_indptr, 1),
            ("kv_seq_lens", kv_seq_lens, 1),
            ("block_tables", block_tables, 2),
        ):
            _expect(isinstance(t, torch.Tensor), f"{name} must be a torch.Tensor")
            _expect(
                t.is_cuda and t.device == self.device,
                f"{name} must be on CUDA device {self.device}, got {t.device}",
            )
            _expect(t.dtype == torch.int32, f"{name} must be int32, got {t.dtype}")
            _expect(
                t.dim() == dim, f"{name} must be {dim}-D, got shape {tuple(t.shape)}"
            )

        b = kv_seq_lens.shape[0]
        _expect(b >= 1, "batch size must be >= 1")
        _expect(
            qo_indptr.shape[0] == b + 1,
            f"qo_indptr must have shape (batch_size+1,) = ({b + 1},), got "
            f"{tuple(qo_indptr.shape)} — it is a token-unit prefix sum "
            "(qo_indptr[0] = 0)",
        )
        _expect(
            block_tables.shape[0] == b,
            f"block_tables must have shape (batch_size, max_pages) with "
            f"batch_size={b}, got {tuple(block_tables.shape)}",
        )
        _expect(
            isinstance(page_size, int) and page_size >= 1,
            f"page_size must be a positive host int, got {page_size!r}",
        )
        for nm, v in (("max_q_len", max_q_len), ("max_kv_len", max_kv_len)):
            _expect(
                isinstance(v, int) and v >= 1,
                f"{nm} must be a positive host int (required; it kills the "
                f"hidden device sync), got {v!r}",
            )
        capacity = block_tables.shape[1] * page_size
        _expect(
            max_kv_len <= capacity,
            f"max_kv_len ({max_kv_len}) exceeds block_tables capacity "
            f"({block_tables.shape[1]} pages x page_size {page_size} = {capacity})",
        )

    def _validate_values(
        self,
        qo_indptr,
        kv_seq_lens,
        block_tables,
        page_size,
        max_q_len,
        max_kv_len,
        qo_indptr_cpu,
        kv_seq_lens_cpu,
    ) -> None:
        """Value-level validation against host mirrors. Always runs.

        These checks are what turns "silently wrong" into "loud error" for
        value corruption: an under-claimed max, an indptr that does not sum
        to the token count, or KV lens exceeding the table capacity would
        otherwise reach a kernel that trusts them as layout/scheduling truth.
        """
        _expect(
            qo_indptr_cpu.device.type == "cpu"
            and tuple(qo_indptr_cpu.shape) == tuple(qo_indptr.shape),
            "qo_indptr_cpu must be a CPU mirror with the same shape as qo_indptr",
        )
        _expect(
            kv_seq_lens_cpu.device.type == "cpu"
            and tuple(kv_seq_lens_cpu.shape) == tuple(kv_seq_lens.shape),
            "kv_seq_lens_cpu must be a CPU mirror with the same shape as kv_seq_lens",
        )
        d = qo_indptr_cpu.diff()
        _expect(bool((d >= 0).all()), "qo_indptr must be non-decreasing")
        _expect(int(qo_indptr_cpu[0]) == 0, "qo_indptr[0] must be 0")
        _expect(
            int(d.max()) <= max_q_len,
            f"max_q_len ({max_q_len}) is smaller than the actual longest "
            f"query ({int(d.max())}) — this would silently corrupt scheduling "
            "or graph shapes downstream",
        )
        _expect(bool((kv_seq_lens_cpu >= 0).all()), "kv_seq_lens must be >= 0")
        _expect(
            int(kv_seq_lens_cpu.max()) <= max_kv_len,
            f"max_kv_len ({max_kv_len}) is smaller than the actual longest "
            f"KV ({int(kv_seq_lens_cpu.max())})",
        )
        capacity = block_tables.shape[1] * page_size
        _expect(
            int(kv_seq_lens_cpu.max()) <= capacity,
            f"kv_seq_lens exceed block_tables capacity ({capacity} tokens)",
        )

    # ------------------------------ run -------------------------------

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _expect(self._planned, "run() called before plan()")
        m = self._meta
        _expect(
            isinstance(kv_cache, tuple) and len(kv_cache) == 2,
            "kv_cache must be a (k_cache, v_cache) tuple of paged tensors "
            "(pages, num_kv_heads, page_size, head_dim) [HND]",
        )
        k_cache, v_cache = kv_cache
        for name, t, d in (
            ("k_cache", k_cache, m["head_dim_qk"]),
            ("v_cache", v_cache, m["head_dim_vo"]),
        ):
            _expect(t.dim() == 4, f"{name} must be 4-D paged (pages, H, page_size, D)")
            _expect(
                t.shape[1] == m["num_kv_heads"]
                and t.shape[2] == m["page_size"]
                and t.shape[3] == d,
                f"{name} shape {tuple(t.shape)} does not match plan "
                f"(H={m['num_kv_heads']}, page_size={m['page_size']}, D={d})",
            )
            _expect(
                t.dtype == m["kv_dtype"],
                f"{name} dtype {t.dtype} != planned {m['kv_dtype']}",
            )
        _expect(
            q.dim() == 3, "q must be packed (total_q_tokens, num_qo_heads, head_dim)"
        )
        _expect(
            q.shape[1] == m["num_qo_heads"] and q.shape[2] == m["head_dim_qk"],
            f"q shape {tuple(q.shape)} does not match plan "
            f"(H={m['num_qo_heads']}, D={m['head_dim_qk']})",
        )
        _expect(q.dtype == m["q_dtype"], f"q dtype {q.dtype} != planned {m['q_dtype']}")
        if m["qo_indptr_cpu"] is not None:
            total = int(m["qo_indptr_cpu"][-1])
            _expect(
                q.shape[0] == total,
                f"q has {q.shape[0]} tokens but qo_indptr sums to {total}",
            )
        cap = CAPABILITIES[self._backend]
        if cap.requires_contiguous_q and not q.is_contiguous():
            raise ValueError(
                f"backend {self._backend!r} requires contiguous packed q "
                "(token-unit addressing assumes packed THD); call "
                ".contiguous() or pin a strided-capable backend (fa2/fa3)"
            )
        if out is not None:
            _expect(
                tuple(out.shape) == (q.shape[0], m["num_qo_heads"], m["head_dim_vo"])
                and out.is_contiguous(),
                "out must be contiguous (total_q_tokens, num_qo_heads, head_dim_vo)",
            )
        return self._adapter.run(q, k_cache, v_cache, out=out, lse=lse)

    def explain(self) -> str:
        _expect(self._planned, "explain() called before plan()")
        return f"chosen: {self._backend}\n{self._resolution.explain()}"


# --------------------------------------------------------------------------
# Derivation layer (proposal §3.2) — library-owned, per-plan, no pointer keying
# --------------------------------------------------------------------------


@dataclass
class _Derived:
    q_seq_lens: torch.Tensor  # (b,)  device — diff(qo_indptr)
    cum_kv_seq_lens: torch.Tensor  # (b+1,) device — for trtllm / cuDNN cu_seq_len_kv
    kv_page_indptr: torch.Tensor  # (b+1,) device CSR page-unit indptr
    kv_page_indices: torch.Tensor  # (total_pages,) device flat page ids
    kv_last_page_len: torch.Tensor  # (b,) device


def _derive(qo_indptr, kv_seq_lens, block_tables, page_size) -> _Derived:
    """Canonical → derived forms.  Pure device ops, zero sync.

    In production this is one fused kernel (proposal §3.2); torch ops keep the
    prototype readable.  The CSR indices buffer is capacity-bounded by the
    block-table width, so sizing needs no sync either.
    """
    zero = torch.zeros(1, dtype=torch.int32, device=kv_seq_lens.device)
    q_seq_lens = qo_indptr.diff()
    cum_kv = torch.cat([zero, torch.cumsum(kv_seq_lens, 0, dtype=torch.int32)])
    pages = (kv_seq_lens + page_size - 1) // page_size  # (b,)
    kv_page_indptr = torch.cat([zero, torch.cumsum(pages, 0, dtype=torch.int32)])
    width = block_tables.shape[1]
    col = torch.arange(width, device=block_tables.device, dtype=torch.int32)
    mask = col.unsqueeze(0) < pages.unsqueeze(1)  # (b, width), row-major keeps order
    kv_page_indices = block_tables[mask].to(torch.int32)
    kv_last_page_len = (kv_seq_lens - 1) % page_size + 1
    return _Derived(
        q_seq_lens, cum_kv, kv_page_indptr, kv_page_indices, kv_last_page_len
    )


# --------------------------------------------------------------------------
# Adapter layer — one per backend; owns dialect conversion AND output
# normalization.  Nothing outside an adapter knows a backend's native format.
# --------------------------------------------------------------------------


class _FaAdapter:
    """fa2/fa3 via the existing BatchPrefillWithPagedKVCacheWrapper.

    Dialect: CSR page metadata + host arrays for the split-KV scheduler.
    With mirrors: zero-sync plan (host CSR computed from the mirrors).
    Without: one documented D2H of (qo_indptr, kv_seq_lens).
    """

    def __init__(self, device, backend: str = "fa2"):
        from ..prefill import BatchPrefillWithPagedKVCacheWrapper

        self._backend = backend
        self._workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, "HND", backend=backend
        )

    def plan(self, m: dict, d: _Derived) -> None:
        if m["qo_indptr_cpu"] is not None and m["kv_seq_lens_cpu"] is not None:
            qo_host = m["qo_indptr_cpu"].to(torch.int32)
            kv_lens_host = m["kv_seq_lens_cpu"].to(torch.int32)
        else:
            # Documented sync (proposal §3.1): fa2/fa3 scheduling needs values.
            qo_host = m["qo_indptr"].cpu()
            kv_lens_host = m["kv_seq_lens"].cpu()
        page = m["page_size"]
        pages_host = (kv_lens_host + page - 1) // page
        kv_indptr_host = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(pages_host, 0, dtype=torch.int32),
            ]
        )
        last_len_host = ((kv_lens_host - 1) % page + 1).to(torch.int32)
        self._wrapper.plan(
            qo_host,
            kv_indptr_host,
            d.kv_page_indices,
            last_len_host,
            m["num_qo_heads"],
            m["num_kv_heads"],
            m["head_dim_qk"],
            page,
            head_dim_vo=m["head_dim_vo"],
            causal=m["causal"],
            sm_scale=m["sm_scale"],
            q_data_type=m["q_dtype"],
            kv_data_type=m["kv_dtype"],
        )
        self._return_lse = m["return_lse"]

    def run(self, q, k_cache, v_cache, *, out=None, lse=None):
        r = self._wrapper.run(
            q, (k_cache, v_cache), out=out, lse=lse, return_lse=self._return_lse
        )
        return r if self._return_lse else (r, None)


class _CudnnAdapter:
    """cuDNN via cudnn_batch_prefill_with_kv_cache (post-#3921 tokens mode).

    Dialect: token-unit indptr as batch offsets (units="tokens"), per-request
    lens as (b,1,1,1).  Native LSE is natural-log padded (b, max_q, h); this
    adapter gathers it to packed (tokens, h) and folds to base-2 — the LSE
    outlier (fragmentation survey A4) dies here, invisibly to callers.
    """

    def __init__(self, device):
        self._workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )

    def plan(self, m: dict, d: _Derived) -> None:
        self._m, self._d = m, d

    def run(self, q, k_cache, v_cache, *, out=None, lse=None):
        from ..cudnn import cudnn_batch_prefill_with_kv_cache

        m, d = self._m, self._d
        b = m["batch_size"]
        native_lse = None
        if m["return_lse"]:
            native_lse = torch.empty(
                b,
                m["max_q_len"],
                m["num_qo_heads"],
                device=q.device,
                dtype=torch.float32,
            )
        out_t, lse_t = cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            m["sm_scale"],
            self._workspace,
            max_token_per_sequence=m["max_q_len"],
            max_sequence_kv=m["max_kv_len"],
            actual_seq_lens_q=d.q_seq_lens.view(b, 1, 1, 1),
            actual_seq_lens_kv=m["kv_seq_lens"].view(b, 1, 1, 1),
            block_tables=m["block_tables"],
            causal=m["causal"],
            return_lse=m["return_lse"],
            batch_offsets_q=m["qo_indptr"],
            batch_offsets_units="tokens",
            out=out,
            lse=native_lse,
        )
        if not m["return_lse"]:
            return out_t, None
        # padded (b, max_q, h) natural-log -> packed (tokens, h) base-2
        total = q.shape[0]
        batch_ids = torch.repeat_interleave(
            torch.arange(b, device=q.device), d.q_seq_lens.to(torch.int64)
        )
        pos = torch.arange(total, device=q.device) - m["qo_indptr"][batch_ids].to(
            torch.int64
        )
        packed = lse_t[batch_ids, pos, :] * _LOG2E
        if lse is not None:
            lse.copy_(packed)
            packed = lse
        return out_t, packed


class _TrtllmGenAdapter:
    """trtllm-gen via trtllm_batch_context_with_kv_cache.

    Dialect: the unified form natively (this is where the canonical form came
    from); the only derivations are cum_kv_seq_lens and the bmm scale fold
    (bmm1 = sm_scale for unquantized, bmm2 = 1.0).  Workspace must be
    zero-initialized (kernel counter semantics).
    """

    def __init__(self, device):
        self._workspace = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )

    def plan(self, m: dict, d: _Derived) -> None:
        self._m, self._d = m, d

    def run(self, q, k_cache, v_cache, *, out=None, lse=None):
        from ..prefill import trtllm_batch_context_with_kv_cache

        m, d = self._m, self._d
        result = trtllm_batch_context_with_kv_cache(
            q,
            (k_cache, v_cache),
            self._workspace,
            m["block_tables"],
            m["kv_seq_lens"],
            m["max_q_len"],
            m["max_kv_len"],
            m["sm_scale"],  # bmm1: sm_scale (q/k descales fold here when quantized)
            1.0,  # bmm2
            m["batch_size"],
            m["qo_indptr"],
            d.cum_kv_seq_lens,
            kv_layout="HND",
            causal=m["causal"],
            out=out,
            lse=lse,
            return_lse=m["return_lse"],
        )
        if m["return_lse"]:
            out_t, lse_t = result
            return out_t, lse_t
        return result, None


_ADAPTERS = {
    "fa2": lambda dev: _FaAdapter(dev, "fa2"),
    "fa3": lambda dev: _FaAdapter(dev, "fa3"),
    "cudnn": _CudnnAdapter,
    "trtllm-gen": _TrtllmGenAdapter,
}
