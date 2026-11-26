# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The delegate behind ``batch_invariant=True`` on ``BatchDecodeWithPagedKVCacheWrapper``, ``BatchPrefillWithPagedKVCacheWrapper`` and
``BatchPrefillWithRaggedKVCacheWrapper``: the wrappers' ``plan`` / ``run`` hand their arguments here, the class's typed checks raise
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for everything the class does not cover (nothing falls back), and the
engine (``engine.plan_attention`` / ``engine.run_attention``) runs the kernels. One class per head dim and mask mode: under the causal
class a token's output bits depend on its KV position, its head and the window alone; under the non-causal class (``causal=False`` on
the prefill wrappers, ``is_causal=False`` with ``q_len_per_req > 1`` on the decode wrapper) on its request's key count and its head
alone. The three wrappers compute the same bits for the same token of the same class.
Chunked prefill: under the causal class a request cut into chunks (each later chunk over the keys cached so far) reproduces the
one-shot bits; under the non-causal class only query chunks planned over the request's full key set do -- a chunk run while the
KV is still growing is that chunk's view of the keys present at the time, and a different ``kv_len`` is a different term.

``run(..., return_lse=True)`` also returns the class's LSE member (``spec.lse_term``): fp32, ``(total_q, num_qo_heads)``, base 2,
or -- on the prefill wrappers -- base e (``lse_base="ln"``, the declared in-kernel multiply by ``fp32(ln 2)``) and the
``(num_qo_heads, total_q)`` layout (``lse_layout="HN"``, a copy of the same bits). It reads the ``(m, l)`` that normalise the
output: the output bits are the same with and without it, and its receipt is its own
(``fingerprint(<wrapper>, head_dim=..., causal=..., lse_base=...)``).

``sinks`` (``run(..., sinks=)`` on the paged wrappers, the ``_sinks`` attribute of the ragged one), an fp32 ``(num_qo_heads,)``
tensor of per-head sink logits, selects the class's sink member (``spec.sink_merge``): head ``h``'s rows get one virtual key
of logit ``sinks[h]`` (natural-log units of the scaled score, ``sm_scale`` not applied) and zero value, merged once after the
row's last chunk; the LSE includes it. A call without sinks is the class without the step, bit for bit; the member's receipts
are ``fingerprint(<wrapper>, head_dim=..., causal=..., sinks=True[, lse_base=...])``. The tensor is read in place at every launch.
A run with sinks and a run without are two members with different bits, so ``plan(..., has_sinks=True | False)`` declares
which one the plan's runs are: a ``run()`` that departs from the declaration raises (``None``, the default, leaves the
choice to each run). The check is made in Python at ``run()``, so capturing ``run()`` into a CUDA graph is unaffected.
The values are not checked. A ``+inf`` or NaN sink gives NaN on its head's rows and a ``-inf`` sink the bits of the run
without sinks (``spec.sink_merge``); the step works on the base-2 value ``mul(sinks[h], LOG2E)``, so a finite sink above
about ``2.36e38`` (the value overflows to ``+inf``) gives NaN as well, and one below about ``-2.36e38`` (``-inf``) the
bits of the run without sinks.

Both members exist in both mask modes: they read a row's final ``(m, l)``, which the non-causal class forms over the request's
whole key range. Under ``causal=False`` (``is_causal=False`` with ``q_len_per_req > 1`` on the decode wrapper) every query row of
a request has the same visible keys, so a row's LSE depends on its request's key count, its head and its own query alone, and
the sink joins that key set; a window stays refused, ``has_sinks`` and the buffer checks are the same, and the receipts take the
mode (``fingerprint(<wrapper>, head_dim=..., causal=False, lse_base=... | sinks=True)``: ``attn-bf16-d<D>-nc-lse-<base>-...``,
``attn-bf16-d<D>-nc-sink-...``, ``attn-bf16-d<D>-nc-sink-lse-<base>-...``). No row of either class is empty (``plan()``
requires ``kv_len >= q_len`` under the causal class and a key for every request with query tokens under the non-causal one),
so no stored LSE is that of an empty row.

CUDA graphs: capture ``run()`` after ``plan()``; a replay reproduces the planned batch (the descriptor table and the launch grid are
the plan's), and a new ``plan()`` needs a new capture. The wrappers' ``use_cuda_graph=True`` contract (``plan()`` refreshes the
metadata a captured graph reads) is not offered: there is no device-side planner, so it raises.

A plan's execution state (the split-KV partials in the workspace, the arrival counters) is single-use per launch: two ``run()``
calls of one plan must be stream-ordered, as with every FlashInfer wrapper's workspace; the counters reset themselves at every
launch, so sequential reuse on one stream needs nothing more.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ...batch_invariant.support import check_device, unsupported
from . import engine
from .spec import HEAD_DIMS, LSE_BASES, scale_log2

CLASS = "attention.bf16"
#: The compute capabilities the class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
ATTENTION_ARCHS = ((10, 0), (10, 3), (10, 7))
_MAX_PAGE = 128


def _dtype(x) -> torch.dtype:
    from ...utils import canonicalize_torch_dtype

    return canonicalize_torch_dtype(x) if x is not None else torch.bfloat16


class BatchInvariantAttention:
    """One instance per wrapper: ``kind`` is ``"decode"`` (paged KV, one or more query tokens per request), ``"prefill"`` (paged
    KV) or ``"ragged"`` (the K / V of every request packed along the token dimension)."""

    CLASS = CLASS

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str,
        kind: str,
        backend: str,
        jit_args,
        use_cuda_graph: bool = False,
    ) -> None:
        assert kind in ("decode", "prefill", "ragged"), kind
        if use_cuda_graph:
            raise unsupported(
                CLASS,
                "use_cuda_graph=True with batch_invariant=True",
                "capturing run() after plan(): a replay reproduces the planned batch, a new plan() needs a new capture",
            )
        if backend != "auto":
            raise unsupported(
                CLASS,
                f'backend="{backend}" with batch_invariant=True',
                'backend="auto" (the mode selects its own kernels)',
            )
        if jit_args is not None:
            raise unsupported(
                CLASS,
                "jit_args with batch_invariant=True",
                "no JIT customisation (the class has one dataflow)",
            )
        if kv_layout not in ("NHD", "HND"):
            raise unsupported(CLASS, f"kv_layout {kv_layout!r}", '"NHD" or "HND"')
        check_device(float_workspace_buffer.device, CLASS, ATTENTION_ARCHS)
        self.kind = kind
        self.kv_layout = kv_layout
        self.device = float_workspace_buffer.device
        self.workspace = float_workspace_buffer
        self.reset_workspace(float_workspace_buffer)
        self._planned: Optional[engine._Planned] = None
        self._config: Optional[engine.AttnConfig] = None
        self._kv_indptr: Optional[torch.Tensor] = None
        self._kv_indices: Optional[torch.Tensor] = None
        self._sm_scale: Optional[float] = None
        self._scale_log2 = 0.0
        self._window_left = -1
        self._causal = True  # the planned class's mask mode
        self._total_q = 0
        self._heads: Tuple[int, int, int] = (0, 0, 0)
        self._page = 1
        self._max_page = -1  # the largest page index the plan references (paged kinds)
        self._total_kv = 0  # the planned packed token count (ragged)

    def reset_workspace(self, float_workspace_buffer: torch.Tensor) -> None:
        """The wrappers' ``reset_workspace_buffer``: the next ``plan()`` sizes its split-KV partials in this buffer (the current plan
        keeps the buffer it was built in)."""
        if float_workspace_buffer.device != self.device:
            raise unsupported(
                CLASS,
                f"a float workspace buffer on {float_workspace_buffer.device}",
                f"a workspace on the wrapper's device {self.device}",
            )
        if float_workspace_buffer.data_ptr() % 16:
            raise unsupported(
                CLASS,
                "a float workspace buffer that is not 16-byte aligned",
                "a 16-byte-aligned workspace",
            )
        self.workspace = float_workspace_buffer

    #: The wrappers' ``plan()`` arguments the class does not take, with what the class takes instead: any of them set is refused.
    _PLAN_ARGS_OUTSIDE_THE_CLASS = {
        "window_right": "window_right 0 (the causal mask)",
        "seq_lens": "the KV lengths of the page table (indptr / last_page_len) or of kv_indptr",
        "seq_lens_q": "the query lengths of qo_indptr",
        "block_tables": "the paged KV CSR triple",
        "prefix_len_ptr": "the causal mask with an optional sliding window (no multi-item scoring)",
        "token_pos_in_items_ptr": "the causal mask with an optional sliding window (no multi-item scoring)",
        "token_pos_in_items_len": "the causal mask with an optional sliding window (no multi-item scoring)",
        "max_item_len_ptr": "the causal mask with an optional sliding window (no multi-item scoring)",
        "v_indptr": "the offsets of kv_indptr for V",
        "o_indptr": "the offsets of qo_indptr for the output",
        "rope_scale": "no fused RoPE (apply RoPE to q / k beforehand)",
        "rope_theta": "no fused RoPE (apply RoPE to q / k beforehand)",
        "positional_arguments": "keyword arguments after the KV cache",
    }

    def refuse(self, **args) -> None:
        """The wrappers call this with the ``plan()`` arguments the class does not take (``name=value``): a value that is set
        (not ``None``, not ``0`` / ``False``) raises the class's typed error before anything is planned."""
        for name, value in args.items():
            if value is None or (isinstance(value, (int, bool)) and not value):
                continue
            raise unsupported(
                CLASS,
                name if isinstance(value, torch.Tensor) else f"{name}={value}",
                self._PLAN_ARGS_OUTSIDE_THE_CLASS.get(name, "the class's own inputs"),
            )

    def workspace_bytes(
        self,
        qo_indptr: Optional[torch.Tensor],
        kv_indptr: torch.Tensor,
        kv_last_page_len: Optional[torch.Tensor],
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        window_left: int = -1,
        q_len_per_req: int = 1,
        *,
        causal: bool = True,
    ) -> int:
        """The wrappers' ``workspace_size``: the float workspace bytes the split-KV plan of this batch needs (0 when every unit is
        fused; a smaller buffer makes ``plan()`` fall back to fused units, which cost parallelism, never bits)."""
        D = self._check_head_dim(head_dim)
        self._check_heads(num_qo_heads, num_kv_heads)
        page_size = self._check_page_size(page_size)
        causal, window_left = self._check_mask_mode(causal, window_left)
        qo_host, kv_host = self._host_offsets(qo_indptr, kv_indptr, q_len_per_req)
        last_host = None
        if self.kind != "ragged":
            if kv_last_page_len is None:
                raise unsupported(
                    CLASS,
                    "a paged plan without kv_last_page_len",
                    "the paged KV CSR triple",
                )
            last_host = self._host_last_page_len(
                kv_last_page_len, kv_host.numel() - 1, page_size
            )
        self._check_requests(qo_host, kv_host, last_host, page_size, causal)
        return engine.workspace_bytes(
            qo_host,
            kv_host,
            last_host,
            num_qo_heads,
            num_kv_heads,
            D,
            page_size,
            window_left,
            engine.default_config(D=D),
            self.device,
            ragged=self.kind == "ragged",
            causal=causal,
        )

    # ------------------------------------------------------------------ the checks plan() and workspace_bytes() share
    @staticmethod
    def _check_head_dim(head_dim: int) -> int:
        D = int(head_dim)
        if D not in HEAD_DIMS:
            raise unsupported(CLASS, f"head_dim {D}", f"head dims {list(HEAD_DIMS)}")
        return D

    @staticmethod
    def _check_heads(num_qo_heads: int, num_kv_heads: int) -> None:
        if (
            num_kv_heads <= 0
            or num_qo_heads < num_kv_heads
            or num_qo_heads % num_kv_heads
        ):
            raise unsupported(
                CLASS,
                f"{num_qo_heads} query heads over {num_kv_heads} KV heads",
                "num_qo_heads a positive multiple of num_kv_heads",
            )
        g = num_qo_heads // num_kv_heads
        if g > 128:
            raise unsupported(
                CLASS, f"a GQA group of {g}", "at most 128 query heads per KV head"
            )

    def _check_page_size(self, page_size: int) -> int:
        """The planner's page size: 1 for the ragged layout, else the caller's (a power of two up to ``_MAX_PAGE``)."""
        if self.kind == "ragged":
            page_size = 1
        if page_size < 1 or page_size > _MAX_PAGE or page_size & (page_size - 1):
            raise unsupported(
                CLASS, f"page_size {page_size}", f"a power of two up to {_MAX_PAGE}"
            )
        return page_size

    @staticmethod
    def _check_mask_mode(causal: bool, window_left: Optional[int]) -> Tuple[bool, int]:
        """The mask mode and the window as the planner takes them: a window belongs to the causal class alone."""
        causal = bool(causal)
        if window_left is None:
            window_left = -1
        window_left = int(window_left)
        if window_left < -1:
            raise unsupported(
                CLASS,
                f"window_left {window_left}",
                "window_left >= 0, or -1 for no window",
            )
        if not causal and window_left >= 0:
            # a symmetric window over a non-causal row would be a further class; the non-causal class sees every key
            raise unsupported(
                CLASS,
                f"window_left {window_left} with causal=False",
                "the whole key range under the non-causal class (a window belongs to the causal class)",
            )
        return causal, window_left

    def _host_offsets(
        self,
        qo_indptr: Optional[torch.Tensor],
        kv_indptr: torch.Tensor,
        q_len_per_req: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The query and KV offsets on the host as int64 (decode: ``q_len_per_req`` query tokens per request), checked for
        the planner: one KV range per request, non-negative offsets below 2^31, query rows from row 0."""
        if self.kind == "decode":
            if q_len_per_req < 1:
                raise unsupported(
                    CLASS,
                    f"q_len_per_req {q_len_per_req}",
                    "at least one query token per request",
                )
            batch = int(kv_indptr.numel()) - 1
            qo_indptr = torch.arange(
                0, batch + 1, dtype=torch.int32, device="cpu"
            ) * int(q_len_per_req)
        else:
            if qo_indptr is None:
                raise unsupported(
                    CLASS, "a plan without qo_indptr", "the packed query offsets"
                )
        kv_host = kv_indptr.to("cpu", non_blocking=False).to(torch.int64)
        qo_host = qo_indptr.to("cpu", non_blocking=False).to(torch.int64)
        if bool((kv_host < 0).any()):
            raise unsupported(
                CLASS,
                f"a negative kv_indptr entry ({int(kv_host.min())})",
                "non-negative KV offsets",
            )
        if qo_host.numel() != kv_host.numel():
            raise unsupported(
                CLASS,
                f"{qo_host.numel() - 1} query requests over {kv_host.numel() - 1} KV requests",
                "one KV range per request",
            )
        if qo_host.numel() and int(qo_host[0]) != 0:
            raise unsupported(
                CLASS,
                f"qo_indptr starting at {int(qo_host[0])}",
                "qo_indptr[0] == 0 (the packed query rows start at row 0)",
            )
        if kv_host.numel() and int(kv_host[-1]) >= 2**31:
            raise unsupported(
                CLASS,
                f"kv_indptr[-1] = {int(kv_host[-1])}",
                "page-table offsets below 2^31 (the kernels read them as int32)",
            )
        return qo_host, kv_host

    @staticmethod
    def _host_last_page_len(
        kv_last_page_len: torch.Tensor, n_requests: int, page_size: int
    ) -> torch.Tensor:
        last_host = kv_last_page_len.to("cpu", non_blocking=False).to(torch.int64)
        if last_host.numel() != n_requests:
            raise unsupported(
                CLASS,
                f"kv_last_page_len of {last_host.numel()} entries for {n_requests} requests",
                "one last-page length per request",
            )
        if bool(((last_host < 0) | (last_host > page_size)).any()):
            raise unsupported(
                CLASS,
                "a kv_last_page_len outside 0..page_size",
                f"last-page lengths in 0..{page_size}",
            )
        return last_host

    @staticmethod
    def _check_requests(
        qo_host: torch.Tensor,
        kv_host: torch.Tensor,
        last_host: Optional[torch.Tensor],
        page_size: int,
        causal: bool,
    ) -> None:
        """Every request's query and key counts against the planned class (``last_host`` is ``None`` for the ragged layout,
        whose ``kv_host`` holds token offsets): the causal class needs ``kv_len >= q_len``, the non-causal class a key for
        every request with query tokens."""
        kv_spans = kv_host[1:] - kv_host[:-1]  # tokens (ragged) or pages
        if last_host is None:
            kv_lens = kv_spans.tolist()
        else:
            # a request without pages has no keys, whatever its last-page entry (the planner's rule)
            kv_lens = torch.where(
                kv_spans > 0, (kv_spans - 1) * page_size + last_host, 0
            ).tolist()
        q_lens = (qo_host[1:] - qo_host[:-1]).tolist()
        for i, (ql, kl) in enumerate(zip(q_lens, kv_lens, strict=True)):
            if ql < 0 or kl < 0:
                raise unsupported(
                    CLASS,
                    f"request {i} with a decreasing qo_indptr / kv_indptr",
                    "non-decreasing offsets",
                )
            if causal and ql > kl:
                raise unsupported(
                    CLASS,
                    f"request {i} with {ql} query tokens over {kl} keys",
                    "kv_len >= q_len per request (the causal class)",
                )
            if not causal and ql > 0 and kl == 0:
                raise unsupported(
                    CLASS,
                    f"request {i} with {ql} query tokens over no key",
                    "at least one key per request with query tokens (a non-causal row sees its request's keys)",
                )

    # ------------------------------------------------------------------ plan
    def plan(
        self,
        *,
        qo_indptr: Optional[torch.Tensor],
        kv_indptr: torch.Tensor,
        kv_indices: Optional[torch.Tensor],
        kv_last_page_len: Optional[torch.Tensor],
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: Optional[int],
        page_size: int,
        causal: bool,
        window_left: int,
        logits_soft_cap: Optional[float],
        pos_encoding_mode: str,
        q_data_type,
        kv_data_type,
        o_data_type,
        sm_scale: Optional[float],
        q_len_per_req: int = 1,
        non_blocking: bool = True,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        use_fp16_qk_reduction: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
        use_token_head_sf: bool = False,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        has_sinks: Optional[bool] = None,
    ) -> None:
        """The wrappers' ``plan`` arguments; every input outside the class raises before any work. ``has_sinks`` declares
        whether every ``run()`` of this plan is given sinks (``True`` / ``False``: a run that departs from it raises);
        ``None`` leaves the choice to each run."""
        D = self._check_head_dim(head_dim_qk)
        if head_dim_vo is not None and int(head_dim_vo) != D:
            raise unsupported(
                CLASS,
                f"head_dim_vo {head_dim_vo} with head_dim_qk {D}",
                "equal query and value head dims",
            )
        self._check_heads(num_qo_heads, num_kv_heads)
        g = num_qo_heads // num_kv_heads
        for name, dt in (
            ("q_data_type", q_data_type),
            ("kv_data_type", kv_data_type),
            ("o_data_type", o_data_type),
        ):
            if _dtype(dt) != torch.bfloat16:
                raise unsupported(
                    CLASS,
                    f"{name} {_dtype(dt)}",
                    "bf16 queries, keys, values and output",
                )
        page_size = self._check_page_size(page_size)
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise unsupported(
                CLASS, f"logits_soft_cap {logits_soft_cap}", "no logits soft cap"
            )
        if pos_encoding_mode not in ("NONE", 0, None):
            raise unsupported(
                CLASS,
                f"pos_encoding_mode {pos_encoding_mode!r}",
                'pos_encoding_mode "NONE" (apply RoPE to q / k beforehand)',
            )
        if rope_scale is not None or rope_theta is not None:
            raise unsupported(CLASS, "rope_scale / rope_theta", "no fused RoPE")
        if custom_mask is not None or packed_custom_mask is not None:
            raise unsupported(
                CLASS,
                "a custom mask",
                "the causal mask with an optional sliding window",
            )
        if use_fp16_qk_reduction:
            raise unsupported(
                CLASS,
                "use_fp16_qk_reduction",
                "the fp32 score accumulation of the class",
            )
        if fixed_split_size is not None or disable_split_kv:
            raise unsupported(
                CLASS,
                "fixed_split_size / disable_split_kv",
                "the class's own split rule (a function of the SM count only)",
            )
        if use_token_head_sf:
            raise unsupported(CLASS, "use_token_head_sf", "unscaled bf16 K / V")
        causal, window_left = self._check_mask_mode(causal, window_left)
        if sm_scale is not None:
            ok = math.isfinite(float(sm_scale)) and float(sm_scale) > 0.0
            if ok:
                try:
                    ok = scale_log2(float(sm_scale)) != 0.0
                except OverflowError:
                    ok = False
            if not ok:
                raise unsupported(
                    CLASS,
                    f"sm_scale {sm_scale}",
                    "a positive finite sm_scale whose log2 scale is a nonzero binary32 number",
                )
        qo_host, kv_indptr_host = self._host_offsets(
            qo_indptr, kv_indptr, q_len_per_req
        )
        last_host = None
        max_page = -1
        if self.kind != "ragged":
            if kv_last_page_len is None or kv_indices is None:
                raise unsupported(
                    CLASS,
                    "a paged plan without kv_indices / kv_last_page_len",
                    "the paged KV CSR triple",
                )
            last_host = self._host_last_page_len(
                kv_last_page_len, kv_indptr_host.numel() - 1, page_size
            )
            n_pages = int(kv_indptr_host[-1]) if kv_indptr_host.numel() else 0
            if kv_indices.numel() < n_pages:
                raise unsupported(
                    CLASS,
                    f"kv_indices of {kv_indices.numel()} entries for kv_indptr[-1] = {n_pages}",
                    "one page index per referenced page",
                )
            if n_pages:
                used = kv_indices[:n_pages]
                bounds = torch.aminmax(used)
                # Only two scalars cross from CUDA; one readback waits for both bounds.
                if used.is_cuda:
                    lo, hi = torch.stack(bounds).to("cpu", non_blocking=False).tolist()
                else:
                    lo, hi = int(bounds.min), int(bounds.max)
                if lo < 0:
                    raise unsupported(
                        CLASS, "a negative page index", "page indices in 0..num_pages-1"
                    )
                max_page = hi  # checked against the cache at run()
        self._check_requests(qo_host, kv_indptr_host, last_host, page_size, causal)
        config = engine.default_config(D=D)
        if (
            D in engine.Q2_DEFAULT_DIMS
            and self.kind != "decode"
            and g >= engine.Q2_DEFAULT_MIN_GROUP[D]
        ):
            from dataclasses import replace

            from .swaps.operands import arch_smem_cap

            # the paired placement (two Q tiles per CTA, one per softmax warpgroup): same bits, faster prefill (paged and ragged) from GQA
            # group engine.Q2_DEFAULT_MIN_GROUP up (smaller groups keep the one-warpgroup body). Its two Q tiles and the
            # smem O_m leave room for a 4-slot K / V ring on 328 KB parts (SM107) and a 3-slot ring on 228 KB parts (SM100 / SM103) at
            # D = 64; at D = 128 the placement fits the 328 KB parts only (engine.Q2_RING)
            slots = engine.Q2_RING[D][arch_smem_cap(self.device) >= (300 << 10)]
            if slots:
                config = replace(config, softmax_wgs=2, q2=True, kv_slots=slots)
        try:
            planned = engine.plan_attention(
                qo_host,
                kv_indptr_host,
                last_host,
                num_qo_heads,
                num_kv_heads,
                D,
                page_size,
                window_left,
                config,
                self.workspace,
                self.device,
                ragged=self.kind == "ragged",
                causal=causal,
                has_sinks=has_sinks,
            )
        except engine.WorkspaceTooSmall:
            # the planner's split units need more partials space than the caller's workspace holds: plan fused units instead (every unit
            # keeps all its chunks in one CTA; the same bits, R3 -- the split kind only buys parallelism), which needs no partials at all
            from dataclasses import replace

            config = replace(config, split_policy="fused")
            try:
                planned = engine.plan_attention(
                    qo_host,
                    kv_indptr_host,
                    last_host,
                    num_qo_heads,
                    num_kv_heads,
                    D,
                    page_size,
                    window_left,
                    config,
                    self.workspace,
                    self.device,
                    ragged=self.kind == "ragged",
                    causal=causal,
                    has_sinks=has_sinks,
                )
            except engine.WorkspaceTooSmall as e:
                raise unsupported(
                    CLASS,
                    f"a float workspace buffer of {self.workspace.numel() * self.workspace.element_size()} bytes",
                    f"at least {e.needed} bytes for this batch's split-KV partials",
                ) from None
        # the kernels read the page table on the device as int32 (CPU metadata, as the other wrappers accept, is staged here) from
        # snapshots: kv_indptr's is the plan's own (engine.plan_attention: the descriptor table holds its offsets), kv_indices' is
        # taken here; a later in-place change by the caller cannot reach a plan built from the old values, and the plan is published
        # only once everything it runs with exists
        kv_indices_dev = (
            kv_indices.to(
                device=self.device,
                dtype=torch.int32,
                non_blocking=non_blocking,
                copy=True,
            ).contiguous()
            if kv_indices is not None
            else torch.zeros(1, dtype=torch.int32, device=self.device)
        )
        self._planned = planned
        self._config = config
        self._kv_indptr = planned.kv_indptr
        self._kv_indices = kv_indices_dev
        self._max_page = max_page
        self._total_kv = int(kv_indptr_host[-1]) if kv_indptr_host.numel() else 0
        self._sm_scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(D)
        self._scale_log2 = scale_log2(self._sm_scale)
        self._window_left = int(window_left)
        self._causal = causal
        self._total_q = int(qo_host[-1].item()) if qo_host.numel() else 0
        self._heads = (num_qo_heads, num_kv_heads, D)
        self._page = page_size

    def validate_forward_options(
        self,
        *,
        sm_scale: Optional[float] = None,
        pos_encoding_mode=None,
        logits_soft_cap: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        window_left: Optional[int] = None,
        causal: Optional[bool] = None,
        use_fp16_qk_reduction: bool = False,
    ) -> None:
        """Check deprecated ``forward()`` overrides without changing the wrapper or its plan."""
        if self._planned is None:
            raise RuntimeError("plan() must be called before forward()")
        effective_scale = (
            sm_scale if sm_scale is not None else 1.0 / math.sqrt(self._heads[2])
        )
        try:
            c = engine.check_sm_scale(effective_scale)
        except ValueError:
            raise unsupported(
                CLASS,
                f"sm_scale {sm_scale}",
                "a positive finite sm_scale whose log2 scale is a nonzero binary32 number",
            ) from None
        # The kernel consumes a binary32 log2 scale, not the caller's Python float.
        if c != self._scale_log2:
            raise unsupported(
                CLASS,
                f"sm_scale {effective_scale} at forward() after planning {self._sm_scale}",
                "the planned binary32 log2 scale",
            )
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise unsupported(
                CLASS, f"logits_soft_cap {logits_soft_cap}", "no logits soft cap"
            )
        if pos_encoding_mode not in ("NONE", 0, None):
            raise unsupported(
                CLASS,
                f"pos_encoding_mode {pos_encoding_mode!r}",
                'pos_encoding_mode "NONE" (apply RoPE to q / k beforehand)',
            )
        self.refuse(rope_scale=rope_scale, rope_theta=rope_theta)
        if causal is not None and bool(causal) != self._causal:
            raise unsupported(
                CLASS,
                f"causal={bool(causal)} at forward() after planning causal={self._causal}",
                "the planned mask mode (the causal and the non-causal class are two classes)",
            )
        if use_fp16_qk_reduction:
            raise unsupported(
                CLASS,
                "use_fp16_qk_reduction",
                "the fp32 score accumulation of the class",
            )
        if window_left is not None and int(window_left) != self._window_left:
            raise unsupported(
                CLASS,
                f"window_left {window_left} at forward() after planning {self._window_left}",
                "the planned window",
            )

    # ------------------------------------------------------------------ run
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: Optional[torch.Tensor],
        return_lse: bool,
        lse: Optional[torch.Tensor],
        window_left: Optional[int],
        scales: Tuple = (),
        sinks=None,
        kv_cache_sf=None,
        skip_softmax_threshold_scale_factor=None,
        enable_pdl: Optional[bool] = None,
        use_fp16_softmax: Optional[bool] = None,
        uses_spcompress: Optional[bool] = None,
        lse_base: str = "log2",
        lse_layout: str = "NH",
    ):
        """Validate changing run-time inputs once, then launch with the fixed plan. ``return_lse``: ``(out, lse)`` with the LSE
        member in ``lse_base`` (``"log2"`` | ``"ln"``) and ``lse_layout`` (``"NH"`` | ``"HN"``), written to ``lse`` when given (as
        on the other backends, an ``lse`` buffer without ``return_lse`` is not used). ``sinks``: the fp32 ``(num_qo_heads,)``
        per-head sink logits of the sink member."""
        if self._planned is None:
            raise RuntimeError("plan() must be called before run()")
        mismatch = engine.sinks_declaration_mismatch(self._planned.has_sinks, sinks)
        if mismatch is not None:
            raise unsupported(
                CLASS,
                mismatch,
                "the same sinks choice at every run() of the plan (a new plan() changes the declaration)",
            )
        if use_fp16_softmax or uses_spcompress:
            raise unsupported(
                CLASS,
                "use_fp16_softmax / uses_spcompress",
                "the fp32 softmax of the class",
            )
        if return_lse:
            if lse_base not in LSE_BASES:
                raise unsupported(
                    CLASS, f"lse_base {lse_base!r}", 'lse_base "log2" or "ln"'
                )
            if lse_layout not in ("NH", "HN"):
                raise unsupported(
                    CLASS, f"lse_layout {lse_layout!r}", 'lse_layout "NH" or "HN"'
                )
        if any(s is not None for s in scales):
            raise unsupported(
                CLASS, "q_scale / k_scale / v_scale / o_scale", "unscaled bf16 operands"
            )
        if kv_cache_sf is not None:
            raise unsupported(CLASS, "a scaled (fp8 / fp4) KV cache", "a bf16 KV cache")
        if skip_softmax_threshold_scale_factor is not None:
            raise unsupported(
                CLASS, "skip_softmax_threshold_scale_factor", "the full softmax"
            )
        if window_left is not None and int(window_left) != self._window_left:
            raise unsupported(
                CLASS,
                f"window_left {window_left} at run() after planning {self._window_left}",
                "the planned window",
            )
        Hq, Hkv, D = self._heads
        # the caller's tensor: the kernels may run on a reshaped or packed copy of it
        q_source = q
        if q.dim() == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
        if q.dim() != 3 or q.shape[1] != Hq or q.shape[2] != D:
            raise unsupported(
                CLASS, f"q of shape {tuple(q.shape)}", f"(total_q, {Hq}, {D})"
            )
        if q.shape[0] != self._total_q:
            raise unsupported(
                CLASS, f"{q.shape[0]} query rows", f"the planned {self._total_q}"
            )
        if (
            q.dtype != torch.bfloat16
            or k.dtype != torch.bfloat16
            or v.dtype != torch.bfloat16
        ):
            raise unsupported(
                CLASS, f"dtypes ({q.dtype}, {k.dtype}, {v.dtype})", "bf16 q, k and v"
            )
        if (
            q.device != self.device
            or k.device != self.device
            or v.device != self.device
        ):
            raise unsupported(
                CLASS,
                f"q / k / v on ({q.device}, {k.device}, {v.device})",
                f"tensors on the wrapper's device {self.device}",
            )
        # the planned page size and head count are what the descriptor table and the KV lengths were built for: a cache of
        # another geometry would be read as live keys with no error from the kernel
        nhd = self.kv_layout == "NHD"
        for name, t in (("k", k), ("v", v)):
            if self.kind == "ragged":
                ok = t.dim() == 3 and (
                    tuple(t.shape[1:]) == (Hkv, D)
                    if nhd
                    else (t.shape[0] == Hkv and t.shape[2] == D)
                )
                want = f"(total_kv, {Hkv}, {D})" if nhd else f"({Hkv}, total_kv, {D})"
                if ok and (t.shape[0] if nhd else t.shape[1]) < self._total_kv:
                    ok = False
                    want += f" with at least the planned {self._total_kv} tokens"
            else:
                inner = (self._page, Hkv, D) if nhd else (Hkv, self._page, D)
                ok = t.dim() == 4 and tuple(t.shape[1:]) == inner
                want = f"(num_pages, {', '.join(map(str, inner))})"
                if ok and t.shape[0] <= self._max_page:
                    ok = False
                    want += f" with more than {self._max_page} pages (the largest planned page index)"
            if ok and t.stride(-1) != 1:
                ok = False
                want += " with a unit stride along the head dim (the kernels are compiled for it)"
            if ok and (t.data_ptr() % 16 or any(s % 8 for s in t.stride()[:-1])):
                ok = False
                want += " 16-byte aligned (base address and outer strides: the TMA descriptors)"
            if not ok:
                raise unsupported(
                    CLASS,
                    f"{name} of shape {tuple(t.shape)} (strides {tuple(t.stride())}) at run()",
                    f"the planned {self.kv_layout} geometry {want}",
                )
        if not q.is_contiguous():
            q = q.contiguous()
        if q.data_ptr() % 16:
            raise unsupported(
                CLASS,
                "a q whose base address is not 16-byte aligned",
                "a 16-byte-aligned q (the TMA descriptor's requirement)",
            )
        storage = engine.plan_storage(self._planned)
        if out is None:
            out = torch.empty_like(q)
        elif (
            out.shape != q.shape
            or out.dtype != torch.bfloat16
            or out.device != self.device
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            raise unsupported(
                CLASS,
                f"out {tuple(out.shape)} {out.dtype} on {out.device}",
                f"a contiguous bf16 ({q.shape[0]}, {Hq}, {D}) output on {self.device}",
            )
        elif engine._output_overlaps_kv(out, k, v):
            raise unsupported(
                CLASS,
                "an output that aliases the KV cache",
                "an output byte span disjoint from the KV caches (CTAs store rows while others still read pages)",
            )
        elif engine._overlaps_any(out, storage):
            raise unsupported(
                CLASS,
                "an output that aliases the float workspace buffer",
                "an output byte span disjoint from the workspace (the launch stores its split-KV partials there)",
            )
        lse_nh = None
        if return_lse:
            nh_shape = (q.shape[0], Hq)
            if lse is not None:
                shape = nh_shape if lse_layout == "NH" else nh_shape[::-1]
                # the page table a run reads is the plan's own copy (plan() snapshots kv_indptr and kv_indices)
                tables = (self._kv_indices, *engine.plan_tables(self._planned))
                bad = engine.check_lse_buffer(
                    lse,
                    q,
                    out,
                    k,
                    v,
                    shape,
                    q_source=q_source,
                    storage=storage,
                    tables=tables,
                )
                if bad is not None:
                    raise unsupported(
                        CLASS,
                        f"an lse buffer {bad}",
                        f"a contiguous fp32 {shape} buffer on {self.device} that aliases neither q, out, the KV cache nor the float workspace buffer",
                    )
            # the kernels write the (total_q, Hq) rows; the HN layout is a copy of them
            lse_nh = (
                lse
                if lse is not None and lse_layout == "NH"
                else torch.empty(nh_shape, dtype=torch.float32, device=self.device)
            )
        if sinks is not None:
            # the HN layout's copy writes the caller's buffer after the launch
            written = (out, lse_nh, lse if return_lse else None, *storage)
            bad = engine.check_sinks(sinks, Hq, self.device, written)
            if bad is not None:
                raise unsupported(
                    CLASS,
                    f"sinks {bad}",
                    f"a contiguous fp32 ({Hq},) tensor of per-head sink logits on {self.device} that aliases neither out, lse nor the float workspace buffer",
                )
        engine._run_attention_validated(
            self._planned,
            q,
            k,
            v,
            self._kv_indptr,
            self._kv_indices,
            self._scale_log2,
            self._window_left,
            self.kv_layout,
            out,
            self._config,
            ragged=self.kind == "ragged",
            enable_pdl=enable_pdl,
            lse=lse_nh,
            lse_mode=engine.LSE_MODES[lse_base] if return_lse else 0,
            sinks=sinks,
        )
        if not return_lse:
            return out
        if lse_layout == "NH":
            return out, lse_nh
        if lse is None:
            lse = torch.empty(nh_shape[::-1], dtype=torch.float32, device=self.device)
        lse.copy_(lse_nh.t())
        return out, lse
