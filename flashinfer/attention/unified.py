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
    query usable at engine init (before pool allocation / graph capture);
    passing the returned ``Resolution`` to ``plan(backend=...)`` pins the
    candidate set — plan() verifies the config matches and may only choose
    within it.
5.  One output contract.  LSE is always base-2 (multiply by ``ln(2)`` to get
    natural log), shape ``(total_q_tokens, num_qo_heads)``, fp32 — adapters
    normalize native formats (cuDNN returns natural-log padded ``(b, max_q,
    h)`` stats; the fold lives in its adapter, not in callers).

Capability honesty rule: ``CAPABILITIES`` declares ONLY what the conformance
matrix and fuzzer actually exercise on hardware.  Production entries are
wider (fa2 head_dim 64/256, trtllm large pages with GQA, NHD layouts, ...);
here an admitted config is a machine-checked config, so under-claiming is the
only honest default.

Prototype simplifications (documented, not hidden):
- dtypes: fp16/bf16 only.  fp8/nvfp4 are capability axes, out of scope here.
- Heuristic order is a static per-arch placeholder, to be seeded from the
  benchmark suite (proposal §5.2).  Autotune hook (§5.4) is not wired.
- CUDA-graph capture mode (pinned metadata buffers, replay-safe re-plan) is
  not wired; run() is capture-shaped (no allocs with out=/lse=, no syncs)
  but the plan-under-capture story is follow-up work.
- ``sinks`` / custom masks / soft-cap are absent capability axes.

Paging metadata comes in exactly one of two forms (never both):
- dense ``block_tables (b, max_pages)`` — vLLM-native; page_size >= 8
  (denser would blow the table up; token-CSR engines use the other form);
- flat ``kv_page_indices`` — sglang-style token/CSR-native, any page_size
  >= 1.  The page-unit indptr and last-page lengths are NOT accepted: they
  are derivable from ``kv_seq_lens`` + ``page_size``, and accepting them
  would create a second truth (see the #3921 mask-divergence class).
  Backends that need the dense table (cudnn, trtllm-gen) get it derived by
  a zero-sync gather when page_size >= 8, and are capability-excluded below
  that.
- Trusted inputs (documented, not validated): host mirrors must match the
  device tensors; ``block_tables`` VALUES (page ids) must be in-pool —
  checking them costs a device-side pass the hot path cannot pay; a debug
  mode (FLASHINFER_VALIDATE_INPUTS-style) is the production answer.
- Stricter than proposal §3.1 in one spot: plan() validates values for every
  backend, so without mirrors even trtllm/cuDNN plans pay the one documented
  D2H (the proposal lets maxes-only suffice there).  Reject-or-correct is
  unconditional in exchange.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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
    sglang's whitelists).  Entries follow the capability-honesty rule from
    the module docstring: declared == machine-checked by the test suites.
    """

    name: str
    # compute-capability majors (minor-insensitive for the prototype)
    cc_majors: frozenset
    q_dtypes: frozenset
    head_dims: frozenset  # of (head_dim_qk, head_dim_vo) pairs
    page_sizes: Optional[frozenset]  # None = any >= the global floor
    kv_layouts: frozenset
    supports_lse: bool
    supports_noncausal: bool
    supports_window: bool
    requires_contiguous_q: bool
    # True if the backend consumes the dense block table (derivation from the
    # flat-indices input form is forbidden below page_size 8 — table blowup)
    needs_dense: bool = False
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
        window_left: int,
        kv_input_form: str,
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
        if window_left >= 0 and not self.supports_window:
            return "sliding window (window_left >= 0) not supported"
        if (
            self.needs_dense
            and kv_input_form == "page_indices"
            and page_size < _MIN_DENSE_PAGE_SIZE
        ):
            return (
                "needs a dense block table, and deriving one from flat page "
                f"indices at page_size {page_size} < {_MIN_DENSE_PAGE_SIZE} "
                "would blow up to (batch, max_context) — CSR-native backends only"
            )
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

# Per the capability-honesty rule: these sets mirror exactly what
# tests/attention/test_unified_prefill_{prototype,fuzzer}.py exercise.
# Production sets are wider (fa2 64/256 head dims, trtllm pages up to 1024
# with GQA per tests/attention/test_trtllm_gen_attention_prefill.py, NHD...).
CAPABILITIES: Dict[str, BackendCapability] = {
    "fa2": BackendCapability(
        name="fa2",
        cc_majors=frozenset({8, 9, 10, 11, 12}),
        q_dtypes=_F16,
        head_dims=frozenset({(64, 64), (128, 128), (256, 256)}),
        page_sizes=None,
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        supports_noncausal=True,
        supports_window=True,
        requires_contiguous_q=False,
    ),
    "fa3": BackendCapability(
        name="fa3",
        cc_majors=frozenset({9}),
        q_dtypes=_F16,
        # (192,128) is NOT declared: the paged fa kernels require
        # k_page_stride == v_page_stride, which separately-allocated
        # K(192)/V(128) pools violate (needs a stride-matched allocation
        # contract; cudnn covers (192,128) without one).
        head_dims=frozenset({(64, 64), (128, 128), (256, 256)}),
        page_sizes=None,
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        supports_noncausal=True,
        supports_window=True,
        requires_contiguous_q=False,
    ),
    "cudnn": BackendCapability(
        name="cudnn",
        cc_majors=frozenset({8, 9, 10, 11, 12}),
        q_dtypes=_F16,
        head_dims=frozenset({(128, 128), (192, 128)}),
        page_sizes=None,
        kv_layouts=frozenset({"HND"}),
        supports_lse=True,
        supports_noncausal=True,  # bottom_right mask off + padding mask: verified H100
        supports_window=False,  # no sliding window in the cuDNN SDPA graph path
        requires_contiguous_q=True,  # token-unit offsets assume packed THD
        needs_dense=True,
        lse_native="ln_padded_bsh",
    ),
    "trtllm-gen": BackendCapability(
        name="trtllm-gen",
        cc_majors=frozenset({10}),
        q_dtypes=_F16,
        head_dims=frozenset({(128, 128)}),
        # 128+ pages are supported by the kernel with GQA (repo tests cover
        # up to 1024) — kept out until this suite exercises them.
        page_sizes=frozenset({16, 32, 64}),
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        # vLLM routes non-causal away from trtllm; unverified here, so False.
        supports_noncausal=False,
        supports_window=True,
        requires_contiguous_q=True,  # TMA: last dim stride 1; keep strict here
        needs_dense=True,
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
    11: ("fa2", "cudnn"),
    12: ("fa2", "cudnn"),
}

# Below this page size a dense (b, max_pages) block table degenerates toward
# (b, max_context): the dense INPUT form requires page_size >= this floor, and
# dense DERIVATION from the flat-indices form is refused below it (backends
# that need the dense table are capability-excluded instead).
_MIN_DENSE_PAGE_SIZE = 8


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


def _resolve_config_key(
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
    window_left: int = -1,
    kv_input_form: str = "block_tables",
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
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) and num_kv_heads ({num_kv_heads}) "
            "must be positive integers"
        )
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads}) for GQA/MQA"
        )
    if kv_input_form not in ("block_tables", "page_indices"):
        raise ValueError(
            f"kv_input_form must be 'block_tables' or 'page_indices', got "
            f"{kv_input_form!r}"
        )
    _expect_page_size(page_size, kv_input_form)

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
            window_left=window_left,
            kv_input_form=kv_input_form,
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
        backends=tuple(candidates),
        excluded=excluded,
        kv_layout=kv_layout,
        config=_resolve_config_key(
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
        ),
    )


# --------------------------------------------------------------------------
# Contract layer
# --------------------------------------------------------------------------


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _expect_page_size(page_size: int, kv_input_form: str) -> None:
    _expect(
        isinstance(page_size, int) and page_size >= 1,
        f"page_size must be a positive host int, got {page_size!r}",
    )
    if kv_input_form == "block_tables":
        _expect(
            page_size >= _MIN_DENSE_PAGE_SIZE,
            f"page_size {page_size} < {_MIN_DENSE_PAGE_SIZE} with the dense "
            "block_tables form: a dense table at token-granular page sizes "
            "degenerates to (batch, max_context) — pass the flat "
            "kv_page_indices form instead (any page_size >= 1)",
        )


class UnifiedPagedPrefill:
    """Unified paged-prefill entry point (prototype).

    Usage (engine-shaped; see ``prototype_demo_unified_prefill.py``)::

        res = resolve_paged_prefill(cc_major=9, num_qo_heads=8, num_kv_heads=2,
                                    head_dim_qk=128, q_dtype=torch.bfloat16,
                                    page_size=16, need_lse=True)
        # ... engine allocates its KV pool in res.kv_layout, sets dtypes ...
        attn = UnifiedPagedPrefill(device)
        attn.plan(qo_indptr=..., kv_seq_lens=..., block_tables=...,
                  page_size=16, max_q_len=..., max_kv_len=...,
                  num_qo_heads=8, num_kv_heads=2, head_dim_qk=128,
                  q_dtype=torch.bfloat16, causal=True, return_lse=True,
                  backend=res)          # pinned candidate set (or a string)
        out, lse = attn.run(q, (k_cache, v_cache))
    """

    def __init__(self, device: Optional[torch.device] = None):
        dev = torch.device(device) if device is not None else torch.device("cuda")
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        self._adapters: Dict[Any, Any] = {}
        self._workspace: Optional[torch.Tensor] = None
        self._planned = False

    def _shared_workspace(self) -> torch.Tensor:
        # One scratch workspace shared by the fa/cudnn adapters (they never
        # run concurrently within one instance).  trtllm-gen keeps a private
        # zero-initialized buffer: its kernels rely on counter semantics that
        # a scribbled-on shared buffer would violate.
        if self._workspace is None:
            self._workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=self.device
            )
        return self._workspace

    # ------------------------------ plan ------------------------------

    def plan(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        block_tables: Optional[torch.Tensor] = None,
        kv_page_indices: Optional[torch.Tensor] = None,
        page_size: int,
        max_q_len: int,
        max_kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: Optional[int] = None,
        q_dtype: torch.dtype,
        kv_dtype: Optional[torch.dtype] = None,
        kv_layout: str = "HND",
        causal: bool = True,
        window_left: int = -1,
        sm_scale: Optional[float] = None,
        return_lse: bool = False,
        qo_indptr_cpu: Optional[torch.Tensor] = None,
        kv_seq_lens_cpu: Optional[torch.Tensor] = None,
        backend: Union[str, Resolution] = "auto",
    ) -> "UnifiedPagedPrefill":
        """Plan one batch with the canonical paged-prefill metadata.

        Canonical metadata (all CUDA int32 on this wrapper's device):

        - ``qo_indptr``: ``(b+1,)`` token-unit prefix sums of query lengths
          (``qo_indptr[0] == 0``; build with ``cumsum(..., dtype=torch.int32)``)
        - ``kv_seq_lens``: ``(b,)`` per-request valid KV lengths >= 1
          (masking truth; causal requires ``q_len_i <= kv_len_i``)
        - paging: EXACTLY ONE of ``block_tables`` ``(b, max_pages_per_seq)``
          dense (page_size >= 8) or ``kv_page_indices`` flat CSR page-id list
          (any page_size >= 1; page-unit indptr / last-page lengths are
          derived from ``kv_seq_lens`` — a second copy would be a second
          truth)
        - ``page_size, max_q_len, max_kv_len``: host ints, REQUIRED
        - ``window_left``: sliding-window size (-1 = unlimited); backends
          without window support are capability-excluded
        - ``qo_indptr_cpu`` / ``kv_seq_lens_cpu``: optional host mirrors —
          with them plan() is zero-sync (validation and fa2/fa3 scheduling
          read the mirrors the engine already owns); without them plan()
          performs ONE documented D2H here.  Value-level validation always
          runs; there is no unvalidated path.
        - ``backend``: a backend name, ``"auto"``, or a ``Resolution`` from
          :func:`resolve_paged_prefill` — the latter pins the candidate set
          decided at engine init (plan() verifies the config matches).
        """
        head_dim_vo = head_dim_vo if head_dim_vo is not None else head_dim_qk
        kv_dtype = kv_dtype if kv_dtype is not None else q_dtype
        _expect(
            kv_layout in ("HND", "NHD"),
            f"kv_layout must be 'HND' or 'NHD', got {kv_layout!r}",
        )
        _expect(
            (block_tables is None) != (kv_page_indices is None),
            "pass EXACTLY ONE paging form: block_tables (dense, vLLM-style) "
            "or kv_page_indices (flat CSR page ids, sglang-style)",
        )
        kv_input_form = "block_tables" if block_tables is not None else "page_indices"

        self._validate_structure(
            qo_indptr,
            kv_seq_lens,
            block_tables,
            kv_page_indices,
            page_size,
            max_q_len,
            max_kv_len,
            kv_input_form,
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
            kv_page_indices,
            page_size,
            max_q_len,
            max_kv_len,
            qo_indptr_cpu,
            kv_seq_lens_cpu,
            causal,
        )

        if isinstance(backend, Resolution):
            # Level-1 pinning (proposal §5.3): verify the plan config matches
            # what the engine resolved at init, then choose within the set.
            want = _resolve_config_key(
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                head_dim_vo,
                q_dtype,
                page_size,
                kv_layout,
                causal,
                return_lse,
                window_left,
                kv_input_form,
            )
            _expect(
                backend.config == want,
                "plan() arguments do not match the pinned Resolution "
                f"(resolved {backend.config}, got {want}) — re-run "
                "resolve_paged_prefill() with the new configuration",
            )
            resolution = backend
        else:
            resolution = resolve_paged_prefill(
                device=self.device,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                q_dtype=q_dtype,
                page_size=page_size,
                kv_layout=kv_layout,
                causal=causal,
                need_lse=return_lse,
                window_left=window_left,
                kv_input_form=kv_input_form,
                backend=backend,
            )
        # Plan-time choice within the pinned set (level 2).  The prototype
        # takes the heuristic head; the autotune hook (proposal §5.4) would
        # consult its cache here, keyed on bucketed (total_q_tokens, max_kv_len).
        self._resolution = resolution
        self._backend = resolution.chosen

        b = kv_seq_lens.shape[0]
        self._meta = dict(
            qo_indptr=qo_indptr,
            kv_seq_lens=kv_seq_lens,
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
            window_left=window_left,
            kv_layout=kv_layout,
            sm_scale=sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim_qk),
            return_lse=return_lse,
            batch_size=b,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_seq_lens_cpu,
        )
        self._derived = _derive(
            qo_indptr,
            kv_seq_lens,
            block_tables,
            kv_page_indices,
            page_size,
            max_kv_len,
            needs_dense=any(CAPABILITIES[n].needs_dense for n in (self._backend,)),
        )
        # both paging forms live in meta post-derivation (None where truly absent)
        self._meta["block_tables"] = (
            block_tables if block_tables is not None else self._derived.block_tables
        )

        key = (self._backend, kv_layout)
        adapter = self._adapters.get(key)
        if adapter is None:
            adapter = _ADAPTERS[self._backend](
                self.device, kv_layout, self._shared_workspace()
            )
            self._adapters[key] = adapter
        adapter.plan(self._meta, self._derived)
        self._adapter = adapter
        self._planned = True
        return self

    def _validate_structure(
        self,
        qo_indptr,
        kv_seq_lens,
        block_tables,
        kv_page_indices,
        page_size,
        max_q_len,
        max_kv_len,
        kv_input_form,
    ) -> None:
        """Closed-set structural validation. Cheap (host-only, shape-derived)."""
        checks = [
            ("qo_indptr", qo_indptr, 1),
            ("kv_seq_lens", kv_seq_lens, 1),
        ]
        if block_tables is not None:
            checks.append(("block_tables", block_tables, 2))
        if kv_page_indices is not None:
            checks.append(("kv_page_indices", kv_page_indices, 1))
        for name, t, dim in checks:
            _expect(isinstance(t, torch.Tensor), f"{name} must be a torch.Tensor")
            _expect(
                t.is_cuda and t.device == self.device,
                f"{name} must be on CUDA device {self.device}, got {t.device}",
            )
            _expect(
                t.dtype == torch.int32,
                f"{name} must be int32, got {t.dtype} — torch cumsum/arange "
                "default to int64; build with dtype=torch.int32 or .int()",
            )
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
        if block_tables is not None:
            _expect(
                block_tables.shape[0] == b,
                f"block_tables must have shape (batch_size, max_pages) with "
                f"batch_size={b}, got {tuple(block_tables.shape)}",
            )
        _expect_page_size(page_size, kv_input_form)
        for nm, v in (("max_q_len", max_q_len), ("max_kv_len", max_kv_len)):
            _expect(
                isinstance(v, int) and v >= 1,
                f"{nm} must be a positive host int (required; it kills the "
                f"hidden device sync), got {v!r}",
            )
        if block_tables is not None:
            capacity = block_tables.shape[1] * page_size
            _expect(
                max_kv_len <= capacity,
                f"max_kv_len ({max_kv_len}) exceeds block_tables capacity "
                f"({block_tables.shape[1]} pages x page_size {page_size} = "
                f"{capacity})",
            )

    def _validate_values(
        self,
        qo_indptr,
        kv_seq_lens,
        block_tables,
        kv_page_indices,
        page_size,
        max_q_len,
        max_kv_len,
        qo_indptr_cpu,
        kv_seq_lens_cpu,
        causal,
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
        if not bool((d > 0).all()):
            bad = int((d <= 0).nonzero()[0])
            raise ValueError(
                f"qo_indptr must be strictly increasing (q_len >= 1); entry "
                f"{bad}->{bad + 1} is {int(qo_indptr_cpu[bad])}->"
                f"{int(qo_indptr_cpu[bad + 1])} — zero-length requests are "
                "outside the v1 envelope; filter them before plan()"
            )
        _expect(int(qo_indptr_cpu[0]) == 0, "qo_indptr[0] must be 0")
        _expect(
            int(d.max()) <= max_q_len,
            f"max_q_len ({max_q_len}) is smaller than the actual longest "
            f"query ({int(d.max())}) — this would silently corrupt scheduling "
            "or graph shapes downstream",
        )
        if not bool((kv_seq_lens_cpu >= 1).all()):
            bad = int((kv_seq_lens_cpu < 1).nonzero()[0])
            raise ValueError(
                f"kv_seq_lens must be >= 1 (request {bad} has "
                f"{int(kv_seq_lens_cpu[bad])}) — zero-length KV rows are "
                "outside the v1 envelope; filter empty requests before plan()"
            )
        if causal and not bool((d <= kv_seq_lens_cpu).all()):
            bad = int((d > kv_seq_lens_cpu).nonzero()[0])
            raise ValueError(
                f"causal masking requires q_len_i <= kv_len_i for every "
                f"request; request {bad} has q_len {int(d[bad])} > kv_len "
                f"{int(kv_seq_lens_cpu[bad])} (fully-masked rows have "
                "backend-divergent LSE semantics and are outside the v1 "
                "envelope)"
            )
        _expect(
            int(kv_seq_lens_cpu.max()) <= max_kv_len,
            f"max_kv_len ({max_kv_len}) is smaller than the actual longest "
            f"KV ({int(kv_seq_lens_cpu.max())})",
        )
        if block_tables is not None:
            capacity = block_tables.shape[1] * page_size
            if not bool((kv_seq_lens_cpu <= capacity).all()):
                bad = int((kv_seq_lens_cpu > capacity).nonzero()[0])
                raise ValueError(
                    f"kv_seq_lens[{bad}] = {int(kv_seq_lens_cpu[bad])} exceeds "
                    f"block_tables capacity ({block_tables.shape[1]} pages x "
                    f"page_size {page_size} = {capacity}) — widen block_tables "
                    "or fix the length"
                )
        else:
            total_pages = int(torch.sum((kv_seq_lens_cpu + page_size - 1) // page_size))
            _expect(
                kv_page_indices.shape[0] >= total_pages,
                f"kv_page_indices has {kv_page_indices.shape[0]} entries but "
                f"kv_seq_lens require {total_pages} pages at page_size "
                f"{page_size} — the flat page-id list must cover "
                "sum(ceil(kv_len/page_size)) entries in request order",
            )

    # ------------------------------ run -------------------------------

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Sequence[torch.Tensor],
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the planned batch.

        - ``q``: packed ``(total_q_tokens, num_qo_heads, head_dim_qk)``,
          dtype as planned.
        - ``kv_cache``: ``(k_cache, v_cache)`` pair, each paged HND
          ``(pages, num_kv_heads, page_size, head_dim)``.
        - ``out``: optional preallocated output, contiguous
          ``(total_q_tokens, num_qo_heads, head_dim_vo)``, dtype == q dtype.
        - ``lse``: optional preallocated LSE buffer, contiguous fp32
          ``(total_q_tokens, num_qo_heads)``; requires ``return_lse=True``.

        Returns ``(out, lse)``; ``lse`` is **base-2** (multiply by ``ln(2)``
        for natural log), packed ``(total_q_tokens, num_qo_heads)``, fp32 —
        identical for every backend.
        """
        _expect(self._planned, "run() called before plan() — call plan() first")
        m = self._meta
        _expect(
            isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2,
            "kv_cache must be a (k_cache, v_cache) pair of paged tensors "
            "(pages, num_kv_heads, page_size, head_dim) [HND]",
        )
        k_cache, v_cache = kv_cache
        layout = m["kv_layout"]
        h_pos, ps_pos = (1, 2) if layout == "HND" else (2, 1)
        shape_word = (
            "(pages, H, page_size, D)"
            if layout == "HND"
            else "(pages, page_size, H, D)"
        )
        for name, t, dim_ in (
            ("k_cache", k_cache, m["head_dim_qk"]),
            ("v_cache", v_cache, m["head_dim_vo"]),
        ):
            _expect(t.dim() == 4, f"{name} must be 4-D paged {shape_word} [{layout}]")
            swapped_hint = (
                f" — dims 1/2 look transposed: the plan declared kv_layout="
                f"{layout!r} {shape_word}; permute(0, 2, 1, 3).contiguous() or "
                "re-plan with the other kv_layout"
                if (
                    t.shape[ps_pos] == m["num_kv_heads"]
                    and t.shape[h_pos] == m["page_size"]
                    and m["num_kv_heads"] != m["page_size"]
                )
                else ""
            )
            _expect(
                t.shape[h_pos] == m["num_kv_heads"]
                and t.shape[ps_pos] == m["page_size"]
                and t.shape[3] == dim_,
                f"{name} shape {tuple(t.shape)} does not match plan "
                f"(kv_layout={layout}, H={m['num_kv_heads']}, "
                f"page_size={m['page_size']}, D={dim_})"
                f"{swapped_hint}",
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
            _expect(
                out.dtype == m["q_dtype"] and out.device == q.device,
                f"out must match q dtype/device ({m['q_dtype']}, {q.device}), "
                f"got ({out.dtype}, {out.device}) — allocate with "
                "torch.empty(..., dtype=q.dtype, device=q.device)",
            )
        if lse is not None:
            _expect(
                m["return_lse"],
                "lse= buffer passed but the plan has return_lse=False — "
                "plan(return_lse=True) or drop the lse= argument",
            )
            _expect(
                tuple(lse.shape) == (q.shape[0], m["num_qo_heads"])
                and lse.dtype == torch.float32
                and lse.is_contiguous()
                and lse.device == q.device,
                "lse must be contiguous fp32 (total_q_tokens, num_qo_heads) "
                f"on {q.device} — the LSE contract is base-2 packed fp32 for "
                "every backend",
            )
        return self._adapter.run(q, k_cache, v_cache, out=out, lse=lse)

    def explain(self) -> str:
        _expect(self._planned, "explain() called before plan() — call plan() first")
        return f"chosen: {self._backend}\n{self._resolution.explain()}"


# --------------------------------------------------------------------------
# Derivation layer (proposal §3.2) — library-owned, per-plan, no pointer keying
# --------------------------------------------------------------------------


@dataclass
class _Derived:
    q_seq_lens: torch.Tensor  # (b,)  device — diff(qo_indptr)
    cum_kv_seq_lens: torch.Tensor  # (b+1,) device — for trtllm / cuDNN cu_seq_len_kv
    kv_page_indptr: torch.Tensor  # (b+1,) device CSR page-unit indptr
    kv_page_indices: torch.Tensor  # flat page ids (CSR-compacted prefix up to
    # kv_page_indptr[-1]; any tail is untouched scratch never read by kernels,
    # which bound reads by the indptr)
    block_tables: Optional[torch.Tensor]  # (b, width) dense — given or derived;
    # None only when no candidate backend needs the dense form


def _derive(
    qo_indptr,
    kv_seq_lens,
    block_tables,
    kv_page_indices,
    page_size,
    max_kv_len,
    *,
    needs_dense: bool,
) -> _Derived:
    """Canonical → derived forms.  Pure device ops, zero sync.

    In production this is one fused kernel (proposal §3.2); torch ops keep the
    prototype readable.  All output shapes are static functions of the input
    shapes and host ints:

    - dense given → flat indices by capacity scatter (a boolean masked-select
      would sync to size its result; scatter does not);
    - flat indices given → dense (when a candidate needs it) by a clamped
      gather of width ceil(max_kv_len / page_size); rows' tails beyond each
      request's page count hold clamped garbage that no kernel reads (they
      bound by kv_seq_lens).
    """
    dev = kv_seq_lens.device
    zero = torch.zeros(1, dtype=torch.int32, device=dev)
    q_seq_lens = qo_indptr.diff()
    cum_kv = torch.cat([zero, torch.cumsum(kv_seq_lens, 0, dtype=torch.int32)])
    pages = (kv_seq_lens + page_size - 1) // page_size  # (b,)
    kv_page_indptr = torch.cat([zero, torch.cumsum(pages, 0, dtype=torch.int32)])
    b = kv_seq_lens.shape[0]

    if block_tables is not None:
        width = block_tables.shape[1]
        capacity = b * width
        col = torch.arange(width, device=dev, dtype=torch.int32)
        valid = col.unsqueeze(0) < pages.unsqueeze(1)  # (b, width)
        # compact destination of each (row, col) lane; invalid lanes all
        # target a dummy tail slot (duplicate writes there are benign)
        dst = kv_page_indptr[:-1].unsqueeze(1).to(torch.int64) + col.unsqueeze(0).to(
            torch.int64
        )
        dst = torch.where(valid, dst, torch.full_like(dst, capacity))
        buf = torch.empty(capacity + 1, dtype=torch.int32, device=dev)
        buf.scatter_(0, dst.reshape(-1), block_tables.reshape(-1))
        return _Derived(
            q_seq_lens, cum_kv, kv_page_indptr, buf[:capacity], block_tables
        )

    dense = None
    if needs_dense:
        width = (max_kv_len + page_size - 1) // page_size  # host int, no sync
        col = torch.arange(width, device=dev, dtype=torch.int64)
        src = kv_page_indptr[:-1].to(torch.int64).unsqueeze(1) + col.unsqueeze(0)
        src = src.clamp_(max=int(kv_page_indices.shape[0]) - 1)
        dense = (
            kv_page_indices.to(torch.int64)
            .gather(0, src.reshape(-1))
            .reshape(b, width)
            .to(torch.int32)
        )
    return _Derived(q_seq_lens, cum_kv, kv_page_indptr, kv_page_indices, dense)


# --------------------------------------------------------------------------
# Adapter layer — one per backend; owns dialect conversion AND output
# normalization.  Nothing outside an adapter knows a backend's native format.
# --------------------------------------------------------------------------


class _FaAdapter:
    """fa2/fa3 via the existing BatchPrefillWithPagedKVCacheWrapper.

    Dialect: CSR page metadata + host arrays for the split-KV scheduler
    (computed from the mirrors the contract layer guarantees — zero-sync).
    """

    def __init__(self, device, kv_layout, workspace, backend: str = "fa2"):
        from ..prefill import BatchPrefillWithPagedKVCacheWrapper

        self._backend = backend
        self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout, backend=backend
        )

    def plan(self, m: dict, d: _Derived) -> None:
        qo_host = m["qo_indptr_cpu"].to(torch.int32)
        kv_lens_host = m["kv_seq_lens_cpu"].to(torch.int32)
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
            window_left=m["window_left"],
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

    def __init__(self, device, kv_layout, workspace):
        assert kv_layout == "HND"  # capability-gated upstream
        self._workspace = workspace.view(torch.int8)

    def plan(self, m: dict, d: _Derived) -> None:
        self._m, self._d = m, d
        # The LSE-gather indices and the native stats buffer are static per
        # plan (qo_indptr and batch size are fixed here) — precompute them so
        # run() stays a single indexed lookup + multiply on the hot path.
        self._native_lse = None
        self._batch_ids = None
        self._pos = None
        if m["return_lse"]:
            dev = m["qo_indptr"].device
            total = int(m["qo_indptr_cpu"][-1])
            token = torch.arange(total, device=dev, dtype=torch.int64)
            bounds = m["qo_indptr"][1:].to(torch.int64)
            self._batch_ids = torch.searchsorted(bounds, token, right=True)
            self._pos = token - m["qo_indptr"].to(torch.int64)[self._batch_ids]
            self._native_lse = torch.empty(
                m["batch_size"],
                m["max_q_len"],
                m["num_qo_heads"],
                device=dev,
                dtype=torch.float32,
            )

    def run(self, q, k_cache, v_cache, *, out=None, lse=None):
        from ..cudnn import cudnn_batch_prefill_with_kv_cache

        m, d = self._m, self._d
        b = m["batch_size"]
        native_lse = self._native_lse
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
        # padded (b, max_q, h) natural-log -> packed (tokens, h) base-2,
        # using the plan-time precomputed gather indices (zero sync).
        packed = lse_t[self._batch_ids, self._pos, :] * _LOG2E
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

    def __init__(self, device, kv_layout, workspace):
        # deliberately NOT the shared workspace: trtllm-gen kernels rely on
        # zero-initialized counter semantics
        self._workspace = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self._kv_layout = kv_layout

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
            window_left=m["window_left"],
            kv_layout=self._kv_layout,
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
    "fa2": lambda dev, layout, ws: _FaAdapter(dev, layout, ws, "fa2"),
    "fa3": lambda dev, layout, ws: _FaAdapter(dev, layout, ws, "fa3"),
    "cudnn": _CudnnAdapter,
    "trtllm-gen": _TrtllmGenAdapter,
}
