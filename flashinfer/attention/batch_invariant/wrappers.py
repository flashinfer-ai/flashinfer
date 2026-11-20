# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The delegate behind ``batch_invariant=True`` on ``BatchDecodeWithPagedKVCacheWrapper``, ``BatchPrefillWithPagedKVCacheWrapper`` and
``BatchPrefillWithRaggedKVCacheWrapper``: the wrappers' ``plan`` / ``run`` hand their arguments here, the class's typed checks raise
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for everything the class does not cover (nothing falls back), and the
engine (``engine.plan_attention`` / ``engine.run_attention``) runs the kernels. One class per head dim: a token's output bits depend on
its KV position, its head and the window alone, so the three wrappers compute the same bits for the same token.

Custom masks (``custom_mask`` / ``packed_custom_mask`` on the two prefill wrappers' ``plan()``) run the explicit-key-set class
(``attention.bf16.mask``): a token's output bits depend on its own mask row (ANDed with the window), its head and its request's
keys alone. The semantics are the regular wrappers': the mask replaces the causal mask (``causal`` is ignored), ``window_left``
is ANDed in, a row without a visible key outputs zeros. The masks are snapshotted at ``plan()``.

``BatchPrefillWithCausalBidirectionalRangesWrapper`` takes the delegate of kind ``"ranges"``: the ranged class, whose rows also see
a key range of their request (``run_ranges``). The ranges arrive at ``run()``, so ``plan()`` checks and keeps the batch metadata
and the unit table is built at the first ``run()`` that brings a given range tensor and window pair (one host readback of the
ranges per call; the tables are kept, so the layers of one forward pass plan once).
That readback makes its ``run()`` an eager call: capturing it in a CUDA graph raises.

CUDA graphs: capture ``run()`` after ``plan()``; a replay reproduces the planned batch (the descriptor table and the launch grid are
the plan's), and a new ``plan()`` needs a new capture. The wrappers' ``use_cuda_graph=True`` contract (``plan()`` refreshes the
metadata a captured graph reads) is not offered: there is no device-side planner, so it raises.

A plan's execution state (the split-KV partials in the workspace, the arrival counters) is single-use per launch: two ``run()``
calls of one plan must be stream-ordered, as with every FlashInfer wrapper's workspace; the counters reset themselves at every
launch, so sequential reuse on one stream needs nothing more.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from ...batch_invariant.support import check_device, unsupported
from . import engine
from .spec import HEAD_DIMS, MASK_CAUSAL, MASK_KEY_RANGES, scale_log2

CLASS = "attention.bf16"
#: unit tables a ranged delegate keeps per plan() (one per distinct range tensor and window pair; the oldest is dropped first)
RANGE_PLANS_KEPT = 8
#: The compute capabilities the class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
ATTENTION_ARCHS = ((10, 0), (10, 3), (10, 7))
_MAX_PAGE = 128


def _dtype(x) -> torch.dtype:
    from ...utils import canonicalize_torch_dtype

    return canonicalize_torch_dtype(x) if x is not None else torch.bfloat16


class BatchInvariantAttention:
    """One instance per wrapper: ``kind`` is ``"decode"`` (paged KV, one or more query tokens per request), ``"prefill"`` (paged
    KV), ``"ragged"`` (the K / V of every request packed along the token dimension) or ``"ranges"`` (paged KV prefill of the
    ranged class: causal plus one bidirectional key range per query token)."""

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
        assert kind in ("decode", "prefill", "ragged", "ranges"), kind
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
        self._total_q = 0
        self._heads: Tuple[int, int, int] = (0, 0, 0)
        self._page = 1
        self._max_page = -1  # the largest page index the plan references (paged kinds)
        self._total_kv = 0  # the planned packed token count (ragged)
        self._masked = False  # the current plan runs the explicit-key-set class
        # kind "ranges": the planned batch's host metadata and the unit tables built from it per (ranges, windows)
        self._range_meta: Optional[dict] = None
        self._range_plans: Dict[tuple, engine._Planned] = {}

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
        # the ranged unit tables are built lazily: the ones built in the old buffer are not reused
        self._range_plans = {}

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
        mask_mode: str = MASK_CAUSAL,
    ) -> int:
        """The wrappers' ``workspace_size``: the float workspace bytes the split-KV plan of this batch needs (0 when every unit is
        fused; a smaller buffer makes ``plan()`` fall back to fused units, which cost parallelism, never bits). ``mask_mode``
        (keyword-only): what a row of the plan will see -- ``"key_set"`` for a plan with a custom mask (the explicit-key-set
        class spans every request's whole key range). Kind ``"ranges"`` (always the ``key_ranges`` mode): the ranges arrive at
        ``run()``, so the size is that of the batch whose every token's range is its request's whole key range -- an upper bound
        for any ranges (the unit count does not depend on them and every split group's chunks are at most these)."""
        D = int(head_dim)
        if D not in HEAD_DIMS:
            raise unsupported(CLASS, f"head_dim {D}", f"head dims {list(HEAD_DIMS)}")
        kv_host = kv_indptr.to("cpu", non_blocking=False).to(torch.int64)
        if self.kind == "decode":
            qo_host = torch.arange(0, kv_host.numel(), dtype=torch.int64) * int(
                q_len_per_req
            )
        else:
            assert qo_indptr is not None
            qo_host = qo_indptr.to("cpu", non_blocking=False).to(torch.int64)
        last_host = (
            kv_last_page_len.to("cpu", non_blocking=False).to(torch.int64)
            if (self.kind != "ragged" and kv_last_page_len is not None)
            else None
        )
        ranges = None
        if self.kind == "ranges":
            assert last_host is not None
            spans = kv_host[1:] - kv_host[:-1]
            kv_lens = torch.where(
                spans > 0, (spans - 1) * int(page_size) + last_host, 0
            )
            hi = torch.repeat_interleave(
                kv_lens, (qo_host[1:] - qo_host[:-1]).clamp(min=0)
            )
            ranges = torch.stack((torch.zeros_like(hi), hi), dim=1)
        return engine.workspace_bytes(
            qo_host,
            kv_host,
            last_host,
            num_qo_heads,
            num_kv_heads,
            D,
            1 if self.kind == "ragged" else page_size,
            -1 if window_left is None else int(window_left),
            engine.default_config(D=D),
            self.device,
            ragged=self.kind == "ragged",
            mask_mode=MASK_KEY_RANGES
            if self.kind == "ranges"
            else (MASK_CAUSAL if self.kind == "decode" else mask_mode),
            ranges=ranges,
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
    ) -> None:
        """The wrappers' ``plan`` arguments; every input outside the class raises before any work."""
        D = int(head_dim_qk)
        if D not in HEAD_DIMS:
            raise unsupported(CLASS, f"head_dim {D}", f"head dims {list(HEAD_DIMS)}")
        if head_dim_vo is not None and int(head_dim_vo) != D:
            raise unsupported(
                CLASS,
                f"head_dim_vo {head_dim_vo} with head_dim_qk {D}",
                "equal query and value head dims",
            )
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
        if self.kind == "ragged":
            page_size = 1
        if page_size < 1 or page_size > _MAX_PAGE or page_size & (page_size - 1):
            raise unsupported(
                CLASS, f"page_size {page_size}", f"a power of two up to {_MAX_PAGE}"
            )
        masked = custom_mask is not None or packed_custom_mask is not None
        if masked and self.kind in ("decode", "ranges"):
            raise unsupported(
                CLASS,
                f"a custom mask on the {self.kind} wrapper",
                "the causal mask with an optional sliding window"
                if self.kind == "decode"
                else "the causal keys and the bidirectional ranges of run()",
            )
        # a custom mask replaces the causal mask, as in the regular kernels: ``causal`` is not read with one.
        # kind "ranges": the wrapper owns its mask (the causal keys and the ranges) and plans with causal=False by contract
        if self.kind in ("prefill", "ragged") and not causal and not masked:
            raise unsupported(
                CLASS,
                "non-causal attention",
                "causal attention (bottom-right aligned), or a custom mask",
            )
        if self.kind == "ranges" and window_left not in (None, -1):
            raise unsupported(
                CLASS,
                f"window_left {window_left} at plan()",
                "the windows of run(causal_window_left=..., range_window_left=...)",
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
        if rope_scale is not None or rope_theta is not None:
            raise unsupported(CLASS, "rope_scale / rope_theta", "no fused RoPE")
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
        if window_left is None:
            window_left = -1
        if window_left < -1:
            raise unsupported(
                CLASS,
                f"window_left {window_left}",
                "window_left >= 0, or -1 for no window",
            )
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
        kv_indptr_host = kv_indptr.to("cpu", non_blocking=False).to(torch.int64)
        qo_host = qo_indptr.to("cpu", non_blocking=False).to(torch.int64)
        if bool((kv_indptr_host < 0).any()):
            raise unsupported(
                CLASS,
                f"a negative kv_indptr entry ({int(kv_indptr_host.min())})",
                "non-negative KV offsets",
            )
        if qo_host.numel() != kv_indptr_host.numel():
            raise unsupported(
                CLASS,
                f"{qo_host.numel() - 1} query requests over {kv_indptr_host.numel() - 1} KV requests",
                "one KV range per request",
            )
        kv_spans = kv_indptr_host[1:] - kv_indptr_host[:-1]  # tokens (ragged) or pages
        if qo_host.numel() and int(qo_host[0]) != 0:
            raise unsupported(
                CLASS,
                f"qo_indptr starting at {int(qo_host[0])}",
                "qo_indptr[0] == 0 (the packed query rows start at row 0)",
            )
        if kv_indptr_host.numel() and int(kv_indptr_host[-1]) >= 2**31:
            raise unsupported(
                CLASS,
                f"kv_indptr[-1] = {int(kv_indptr_host[-1])}",
                "page-table offsets below 2^31 (the kernels read them as int32)",
            )
        last_host = None
        max_page = -1
        if self.kind == "ragged":
            kv_lens = kv_spans.tolist()
        else:
            if kv_last_page_len is None or kv_indices is None:
                raise unsupported(
                    CLASS,
                    "a paged plan without kv_indices / kv_last_page_len",
                    "the paged KV CSR triple",
                )
            last_host = kv_last_page_len.to("cpu", non_blocking=False).to(torch.int64)
            if last_host.numel() != kv_spans.numel():
                raise unsupported(
                    CLASS,
                    f"kv_last_page_len of {last_host.numel()} entries for {kv_spans.numel()} requests",
                    "one last-page length per request",
                )
            if bool(((last_host < 0) | (last_host > page_size)).any()):
                raise unsupported(
                    CLASS,
                    "a kv_last_page_len outside 0..page_size",
                    f"last-page lengths in 0..{page_size}",
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
            if masked:
                if ql > 0 and kl == 0:
                    raise unsupported(
                        CLASS,
                        f"request {i} with {ql} query tokens over no key",
                        "at least one key per request with query tokens (a custom mask selects among the request's keys)",
                    )
            elif ql > kl:
                raise unsupported(
                    CLASS,
                    f"request {i} with {ql} query tokens over {kl} keys",
                    "kv_len >= q_len per request (the causal class)",
                )
        key_sets = None
        if masked:
            # the rows' key sets in the kernels' layout, from a snapshot of the caller's mask (a later in-place change cannot reach
            # the plan); the window is ANDed into the sets here
            for name, t in (
                ("custom_mask", custom_mask),
                ("packed_custom_mask", packed_custom_mask),
            ):
                if t is not None and not isinstance(t, torch.Tensor):
                    raise unsupported(
                        CLASS, f"{name} of type {type(t).__name__}", "a torch.Tensor"
                    )
            try:
                key_sets = engine.pack_key_sets(
                    q_lens,
                    kv_lens,
                    custom_mask if packed_custom_mask is None else None,
                    packed_custom_mask,
                    window_left,
                )
            except ValueError as e:
                raise unsupported(
                    CLASS,
                    str(e),
                    "a custom_mask of sum(q_len * kv_len) elements, or its segment_packbits(bitorder='little') bytes",
                ) from None
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
        range_meta = None
        planned = None
        if self.kind == "ranges":
            # the unit table needs the ranges, which arrive at run(): keep what it is built from (host snapshots)
            q_len_t = qo_host[1:] - qo_host[:-1]
            kv_len_t = torch.tensor(kv_lens, dtype=torch.int64)
            first = kv_len_t - q_len_t  # the position of a request's first query token
            tok = torch.arange(int(qo_host[-1]) if qo_host.numel() else 0)
            range_meta = {
                "qo": qo_host,
                "kv": kv_indptr_host,
                "last": last_host,
                "pos": tok
                - torch.repeat_interleave(qo_host[:-1], q_len_t)
                + torch.repeat_interleave(first, q_len_t),
                "kv_len": torch.repeat_interleave(kv_len_t, q_len_t),
            }
        else:
            planned, config = self._plan_units(
                qo_host,
                kv_indptr_host,
                last_host,
                num_qo_heads,
                num_kv_heads,
                D,
                page_size,
                window_left,
                config,
                key_sets=key_sets,
            )
        # the kernels read the page table on the device as int32 (CPU metadata, as the other wrappers accept, is staged here) from
        # snapshots: kv_indptr's is the plan's own (engine.plan_attention: the descriptor table holds its offsets; kind "ranges"
        # builds its plans at run()), kv_indices' is taken here; a later in-place change by the caller cannot reach a plan built
        # from the old values, and the plan is published only once everything it runs with exists
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
        self._range_meta = range_meta
        self._range_plans = {}
        self._config = config
        self._kv_indptr = planned.kv_indptr if planned is not None else None
        self._kv_indices = kv_indices_dev
        self._max_page = max_page
        self._total_kv = int(kv_indptr_host[-1]) if kv_indptr_host.numel() else 0
        self._sm_scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(D)
        self._scale_log2 = scale_log2(self._sm_scale)
        self._window_left = int(window_left)
        self._total_q = int(qo_host[-1].item()) if qo_host.numel() else 0
        self._heads = (num_qo_heads, num_kv_heads, D)
        self._page = page_size
        self._masked = masked

    def _plan_units(
        self,
        qo_host: torch.Tensor,
        kv_indptr_host: torch.Tensor,
        last_host: Optional[torch.Tensor],
        num_qo_heads: int,
        num_kv_heads: int,
        D: int,
        page_size: int,
        window_left: int,
        config: engine.AttnConfig,
        *,
        key_sets: Optional[engine.KeySets] = None,
        ranges: Optional[torch.Tensor] = None,
    ) -> Tuple[engine._Planned, engine.AttnConfig]:
        """The engine's plan in the wrapper's workspace, and the configuration it was built with. ``key_sets`` (the ``key_set``
        mode) or ``ranges`` (the ``key_ranges`` mode) select what a row sees; neither: the causal mode."""
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
                key_sets=key_sets,
                ranges=ranges,
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
                    key_sets=key_sets,
                    ranges=ranges,
                )
            except engine.WorkspaceTooSmall as e:
                raise unsupported(
                    CLASS,
                    f"a float workspace buffer of {self.workspace.numel() * self.workspace.element_size()} bytes",
                    f"at least {e.needed} bytes for this batch's split-KV partials",
                ) from None
        return planned, config

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
        if (
            causal is not None
            and not causal
            and self.kind != "decode"
            and not self._masked
        ):
            raise unsupported(
                CLASS,
                "non-causal attention",
                "causal attention (bottom-right aligned), or a custom mask",
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
    ) -> torch.Tensor:
        """Validate changing run-time inputs once, then launch with the fixed plan."""
        if self._planned is None:
            raise RuntimeError("plan() must be called before run()")
        self._refuse_run_options(
            return_lse=return_lse,
            lse=lse,
            scales=scales,
            sinks=sinks,
            kv_cache_sf=kv_cache_sf,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            use_fp16_softmax=use_fp16_softmax,
            uses_spcompress=uses_spcompress,
        )
        if window_left is not None and int(window_left) != self._window_left:
            raise unsupported(
                CLASS,
                f"window_left {window_left} at run() after planning {self._window_left}",
                "the planned window",
            )
        q, out = self._checked_run_tensors(q, k, v, out)
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
        )
        return out

    def _refuse_run_options(
        self,
        *,
        return_lse: bool,
        lse: Optional[torch.Tensor],
        scales: Tuple = (),
        sinks=None,
        kv_cache_sf=None,
        skip_softmax_threshold_scale_factor=None,
        use_fp16_softmax: Optional[bool] = None,
        uses_spcompress: Optional[bool] = None,
    ) -> None:
        """The ``run()`` options outside the class: any of them set raises the typed error."""
        if use_fp16_softmax or uses_spcompress:
            raise unsupported(
                CLASS,
                "use_fp16_softmax / uses_spcompress",
                "the fp32 softmax of the class",
            )
        if return_lse or lse is not None:
            raise unsupported(
                CLASS,
                "return_lse",
                "the attention output alone (the class has no LSE member yet)",
            )
        if any(s is not None for s in scales):
            raise unsupported(
                CLASS, "q_scale / k_scale / v_scale / o_scale", "unscaled bf16 operands"
            )
        if sinks is not None:
            raise unsupported(CLASS, "attention sinks", "plain softmax")
        if kv_cache_sf is not None:
            raise unsupported(CLASS, "a scaled (fp8 / fp4) KV cache", "a bf16 KV cache")
        if skip_softmax_threshold_scale_factor is not None:
            raise unsupported(
                CLASS, "skip_softmax_threshold_scale_factor", "the full softmax"
            )

    def _checked_run_tensors(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``run()``'s tensors checked against the plan: the packed contiguous ``q`` and the output to write."""
        Hq, Hkv, D = self._heads
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
        return q, out

    # ------------------------------------------------------------------ run (the ranged class)
    def _class_ranges(self, host: torch.Tensor, range_window: int) -> torch.Tensor:
        """The wrapper's range rows -- inclusive ``[start, end]`` per query token, ``(-1, -1)`` for none, the range part kept
        while ``pos - key < range_window`` when ``range_window > 0`` -- as the class's ranges: the keys ``[a, b)`` of the request
        (``0 <= a <= b <= kv_len``), ``(0, 0)`` for a token without one."""
        assert self._range_meta is not None
        pos, kv_len = self._range_meta["pos"], self._range_meta["kv_len"]
        host = host.to(torch.int64)
        a = host[:, 0].clamp(min=0)
        b = torch.minimum(host[:, 1] + 1, kv_len)
        if range_window > 0:
            a = torch.maximum(a, pos - range_window + 1)
        none = a >= b
        return torch.stack((a.masked_fill(none, 0), b.masked_fill(none, 0)), dim=1)

    def run_ranges(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bidirectional_ranges: torch.Tensor,
        *,
        causal_window_left: int,
        range_window_left: int,
        out: Optional[torch.Tensor],
        return_lse: bool,
        lse: Optional[torch.Tensor],
        scales: Tuple = (),
        kv_cache_sf=None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        """``BatchPrefillWithCausalBidirectionalRangesWrapper.run``: token ``i`` sees its causal keys (bounded by
        ``causal_window_left`` as FlashInfer's ``window_left``; negative: unbounded) and the keys of its range row. The unit
        table of a range tensor and window pair is built at its first call and kept until the next ``plan()``."""
        if self.kind != "ranges" or self._range_meta is None:
            raise RuntimeError("plan() must be called before run()")
        if torch.cuda.is_current_stream_capturing():
            raise unsupported(
                CLASS,
                "capturing run() of the ranges wrapper in a CUDA graph",
                "an eager run() (it reads the range tensor back to the host to build the work units)",
            )
        self._refuse_run_options(
            return_lse=return_lse, lse=lse, scales=scales, kv_cache_sf=kv_cache_sf
        )
        causal_window = -1 if int(causal_window_left) < 0 else int(causal_window_left)
        range_window = max(int(range_window_left), 0)
        q, out = self._checked_run_tensors(q, k, v, out)
        if (
            bidirectional_ranges.dim() != 2
            or tuple(bidirectional_ranges.shape) != (self._total_q, 2)
            or bidirectional_ranges.dtype != torch.int32
        ):
            raise unsupported(
                CLASS,
                f"bidirectional_ranges {tuple(bidirectional_ranges.shape)} {bidirectional_ranges.dtype}",
                f"an int32 ({self._total_q}, 2) tensor: one [start, end] row per planned query token",
            )
        host = bidirectional_ranges.to("cpu", non_blocking=False)
        key = (causal_window, range_window, host.numpy().tobytes())
        planned = self._range_plans.get(key)
        if planned is None:
            Hq, Hkv, D = self._heads
            meta = self._range_meta
            assert self._config is not None
            planned, _ = self._plan_units(
                meta["qo"],
                meta["kv"],
                meta["last"],
                Hq,
                Hkv,
                D,
                self._page,
                causal_window,
                self._config,
                ranges=self._class_ranges(host, range_window),
            )
            while len(self._range_plans) >= RANGE_PLANS_KEPT:
                self._range_plans.pop(next(iter(self._range_plans)))
            self._range_plans[key] = planned
        engine._run_attention_validated(
            planned,
            q,
            k,
            v,
            planned.kv_indptr,
            self._kv_indices,
            self._scale_log2,
            causal_window,
            self.kv_layout,
            out,
            planned.config,
            ragged=False,
            enable_pdl=enable_pdl,
        )
        return out
