# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The delegate behind ``batch_invariant=True`` on ``BatchDecodeWithPagedKVCacheWrapper``, ``BatchPrefillWithPagedKVCacheWrapper`` and
``BatchPrefillWithRaggedKVCacheWrapper``: the wrappers' ``plan`` / ``run`` hand their arguments here, the class's typed checks raise
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for everything the class does not cover (nothing falls back), and the
engine (``engine.plan_attention`` / ``engine.run_attention``) runs the kernels. One class per head dim: a token's output bits depend on
its KV position, its head and the window alone, so the three wrappers compute the same bits for the same token.

``run(..., return_lse=True)`` also returns the class's LSE member (``spec.lse_term``): fp32, ``(total_q, num_qo_heads)``, base 2,
or -- on the prefill wrappers -- base e (``lse_base="ln"``, the declared in-kernel multiply by ``fp32(ln 2)``) and the
``(num_qo_heads, total_q)`` layout (``lse_layout="HN"``, a copy of the same bits). It reads the ``(m, l)`` that normalise the
output: the output bits are the same with and without it, and its receipt is its own
(``fingerprint(<wrapper>, head_dim=..., lse_base=...)``).

``sinks`` (``run(..., sinks=)`` on the paged wrappers, the ``_sinks`` attribute of the ragged one), an fp32 ``(num_qo_heads,)``
tensor of per-head sink logits, selects the class's sink member (``spec.sink_merge``): head ``h``'s rows get one virtual key
of logit ``sinks[h]`` (natural-log units of the scaled score, ``sm_scale`` not applied) and zero value, merged once after the
row's last chunk; the LSE includes it. A call without sinks is the class without the step, bit for bit; the member's receipts
are ``fingerprint(<wrapper>, head_dim=..., sinks=True[, lse_base=...])``. The tensor is read in place at every launch.
A run with sinks and a run without are two members with different bits, so ``plan(..., has_sinks=True | False)`` declares
which one the plan's runs are: a ``run()`` that departs from the declaration raises (``None``, the default, leaves the
choice to each run). The check is made in Python at ``run()``, so capturing ``run()`` into a CUDA graph is unaffected.
The values are not checked: a sink whose base-2 value ``mul(sinks[h], LOG2E)`` overflows binary32 (a magnitude above about
``2.36e38``) is a sink of that infinity, and a ``+inf`` or NaN sink gives NaN on its head's rows (``spec.sink_merge``).

CUDA graphs: capture ``run()`` after ``plan()``; a replay reproduces the planned batch (the descriptor table and the launch grid are
the plan's), and a new ``plan()`` needs a new capture -- unless the wrapper is the decode wrapper built with
``use_cuda_graph=True``, whose contract is the regular wrappers': the batch size and ``q_len_per_req`` are fixed for the wrapper's
lifetime, as are the head counts, the head dim, the page size, the window, ``sm_scale`` and, from the first ``run()``, whether
the LSE and sinks are asked for; ``plan()`` (or ``flashinfer.decode.fast_decode_plan``) before every replay refreshes the KV
lengths and the page table, which a captured ``run()`` then follows. The launch of such a wrapper is fixed at its first plan
(``engine.FixedLaunch``: the kernel, the grid, the tables' addresses and capacities); every plan is written into the same
tables and held to their capacity by fusing units, a placement like every other planner choice, so a replay's bits are those of
an eager call on the same batch. A request without pages (a padded slot) is a null request there: its output rows are zeros and
its LSE ``-inf``, with or without sinks.

A plan's table copies are stream-ordered work of the stream ``plan()`` is called on (non-blocking unless ``non_blocking=False``):
a replay on that stream follows them by itself; a replay on another stream must wait for it (``wait_stream``), and the next
plan for the replay, as with the other backends' metadata copies. Wrappers may share one float workspace: it holds only the
split-KV partials of the launch that is running.

The paged prefill wrapper takes ``use_cuda_graph=True`` the same way, with the regular wrapper's contract: the number of request
slots is fixed and the total query rows stay within a capacity (``max_total_num_rows``, or the first plan's rows); every plan
may give every slot another query length (zero: a slot without rows, the padding of a captured batch) and another KV length. Its
fixed launch is sized for any split of the row capacity over the slots; its kernel is the row body for every plan -- with the
paired placement's kernel where that is the wrapper's default, a batch too small to pair running every unit alone in its pair
-- since the batch can no longer choose the body. ``run()`` is captured on the caller's ``q`` / ``out`` (/ ``lse``) of the
capture, which are the tensors of every replay: a plan's rows are their first ``qo_indptr[-1]`` rows, and no row beyond is
written (with ``lse_layout="HN"`` the transposed copy covers the whole buffer). A request with fewer keys than query tokens is
refused as in eager mode, with no key at all too. The page indices are range-checked at the first plan only (a check is a
device readback, and later plans run before every replay). The ragged prefill wrapper refuses ``use_cuda_graph=True``.

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
        if use_cuda_graph and kind == "ragged":
            raise unsupported(
                CLASS,
                "use_cuda_graph=True with batch_invariant=True on the ragged prefill wrapper",
                "capturing run() after plan(): a replay reproduces the planned batch, a new plan() needs a new capture "
                "(the paged decode and prefill wrappers take use_cuda_graph=True)",
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
        # use_cuda_graph=True (decode): the caller's persistent page-table buffers, the fixed launch of the first plan and
        # what the first run() asked for
        self.graph = bool(use_cuda_graph)
        self._kv_buffers: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            None
        )
        self._fixed: Optional[engine.FixedLaunch] = None
        self._frozen_plan: Optional[tuple] = None
        self._frozen_run: Optional[tuple] = None
        self._has_sinks: Optional[bool] = (
            None  # the sinks declaration of a use_cuda_graph=True wrapper's plans
        )
        self._qo_buffer: Optional[torch.Tensor] = None
        self._max_rows: Optional[int] = (
            None  # the packed query rows a plan may hold (prefill)
        )

    def bind_graph_buffers(
        self,
        paged_kv_indptr_buffer: torch.Tensor,
        paged_kv_indices_buffer: torch.Tensor,
        paged_kv_last_page_len_buffer: torch.Tensor,
        qo_indptr_buffer: Optional[torch.Tensor] = None,
        max_total_num_rows: Optional[int] = None,
    ) -> None:
        """The wrappers' page-table buffers under ``use_cuda_graph=True`` (``paged_kv_*_buffer`` on the decode wrapper,
        ``*_buf`` on the prefill wrapper, with its ``qo_indptr_buf`` and ``max_total_num_rows``): ``plan()`` refreshes them
        and the captured kernels read the KV offsets and page indices in place, so they are int32 tensors on the wrapper's
        device."""
        assert self.graph
        self._qo_buffer = qo_indptr_buffer
        self._max_rows = max_total_num_rows
        for name, t in (
            ("the KV indptr buffer", paged_kv_indptr_buffer),
            ("the KV page-index buffer", paged_kv_indices_buffer),
            ("the KV last-page-length buffer", paged_kv_last_page_len_buffer),
            ("the query indptr buffer", qo_indptr_buffer),
        ):
            if t is None:
                continue
            if (
                t.dtype != torch.int32
                or t.device != self.device
                or t.dim() != 1
                or not t.is_contiguous()
            ):
                raise unsupported(
                    CLASS,
                    f"{name} of dtype {t.dtype}, shape {tuple(t.shape)} on {t.device}",
                    f"a contiguous one-dimensional int32 buffer on {self.device}",
                )
        self._kv_buffers = (
            paged_kv_indptr_buffer,
            paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer,
        )

    def reset_workspace(self, float_workspace_buffer: torch.Tensor) -> None:
        """The wrappers' ``reset_workspace_buffer``: the next ``plan()`` sizes its split-KV partials in this buffer (the current plan
        keeps the buffer it was built in)."""
        if float_workspace_buffer.device != self.device:
            raise unsupported(
                CLASS,
                f"a float workspace buffer on {float_workspace_buffer.device}",
                f"a workspace on the wrapper's device {self.device}",
            )
        if getattr(self, "_fixed", None) is not None:
            raise unsupported(
                CLASS,
                "a new float workspace buffer after the first plan() of a use_cuda_graph=True wrapper",
                "the workspace the fixed launch was built in (a captured graph holds its address)",
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
    ) -> int:
        """The wrappers' ``workspace_size``: the float workspace bytes the split-KV plan of this batch needs (0 when every unit is
        fused; a smaller buffer makes ``plan()`` fall back to fused units, which cost parallelism, never bits)."""
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
        seq_lens: Optional[torch.Tensor] = None,
        kv_indptr_host: Optional[torch.Tensor] = None,
        fast: bool = False,
        has_sinks: Optional[bool] = None,
    ) -> None:
        """The wrappers' ``plan`` arguments; every input outside the class raises before any work. ``has_sinks`` declares
        whether every ``run()`` of this plan is given sinks (``True`` / ``False``: a run that departs from it raises);
        ``None`` leaves the choice to each run. ``disable_split_kv`` plans
        fused units and ``fixed_split_size`` of ``None`` / ``-1`` is the wrappers' "no fixed size" (both placements); a positive
        ``fixed_split_size`` is refused -- the chunk length is a class constant -- except under ``use_cuda_graph=True``, where
        it is accepted and not used, as serving engines pass it on every replan. ``seq_lens`` (decode) must be the KV
        lengths of the page table. ``fast`` (``flashinfer.decode.fast_decode_plan``): the replan of a ``use_cuda_graph=True``
        wrapper without device readbacks -- ``kv_indptr_host`` is the host copy of ``kv_indptr`` when given, every last-page
        length is 1 at page size 1 (as that function assumes), and the page indices are not range-checked."""
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
        if self.kind != "decode" and not causal:
            raise unsupported(
                CLASS, "non-causal attention", "causal attention (bottom-right aligned)"
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
        if (
            fixed_split_size is not None
            and int(fixed_split_size) > 0
            and not self.graph
        ):
            raise unsupported(
                CLASS,
                f"fixed_split_size={fixed_split_size}",
                "the class's own chunk length (a class constant); None or -1",
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
            if self.graph:
                # the regular wrappers' frozen shape, with their errors
                if self._kv_buffers is None:
                    raise RuntimeError(
                        "use_cuda_graph=True needs the wrapper's paged_kv_*_buffer tensors"
                    )
                fixed_batch = self._kv_buffers[2].numel()
                if batch != fixed_batch:
                    raise ValueError(
                        "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                        " mismatches the batch size set during initialization {}".format(
                            batch, fixed_batch
                        )
                    )
                if kv_indices is not None and kv_indices.numel() > (
                    self._kv_buffers[1].numel()
                ):
                    raise ValueError(
                        "The size of indices should be less than or equal to the allocated buffer"
                    )
                if self._fixed is not None and (
                    self._fixed.q_len_per_req != int(q_len_per_req)
                ):
                    raise ValueError(
                        "q_len_per_req is part of the frozen cudagraph shape: "
                        f"this wrapper was planned with {self._fixed.q_len_per_req}, got "
                        f"{q_len_per_req}. Use a separate wrapper per q_len_per_req."
                    )
            qo_indptr = torch.arange(
                0, batch + 1, dtype=torch.int32, device="cpu"
            ) * int(q_len_per_req)
        else:
            if qo_indptr is None:
                raise unsupported(
                    CLASS, "a plan without qo_indptr", "the packed query offsets"
                )
            if self.graph:
                # the regular wrapper's frozen shape, with its errors
                if self._kv_buffers is None:
                    raise RuntimeError(
                        "use_cuda_graph=True needs the wrapper's page-table buffers"
                    )
                batch = int(qo_indptr.numel()) - 1
                fixed_batch = self._kv_buffers[2].numel()
                if batch != fixed_batch:
                    raise ValueError(
                        "The batch size should be fixed during the lifecycle of the wrapper in "
                        "cuda graph mode, the runtime batch size {} mismatches the batch size {} "
                        " set during initialization.".format(batch, fixed_batch)
                    )
                if kv_indices is not None and kv_indices.numel() > (
                    self._kv_buffers[1].numel()
                ):
                    raise ValueError(
                        "The length of paged_kv_indices exceeds the allocated buffer size."
                    )
        kv_indptr_host = (
            (kv_indptr if kv_indptr_host is None else kv_indptr_host)
            .to("cpu", non_blocking=False)
            .to(torch.int64)
        )
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
        if (
            self.graph
            and self.kind == "prefill"
            and self._max_rows is not None
            and qo_host.numel()
            and int(qo_host[-1]) > self._max_rows
        ):
            raise ValueError(
                "The total number of rows in qo_indptr {} in cuda graph mode cannot "
                "exceed the number of rows set during initialization {}.".format(
                    int(qo_host[-1]), self._max_rows
                )
            )
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
            if fast and page_size == 1:
                last_host = torch.ones(kv_spans.numel(), dtype=torch.int64)
            else:
                last_host = kv_last_page_len.to("cpu", non_blocking=False).to(
                    torch.int64
                )
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
            # a prefill wrapper under use_cuda_graph checks the page indices at its first plan only: later plans run before
            # every replay and a range check is a device readback
            unchecked = fast or (
                self.graph and self.kind == "prefill" and self._fixed is not None
            )
            if n_pages and not unchecked:
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
            if ql > kl and not (self.graph and self.kind == "decode" and kl == 0):
                raise unsupported(
                    CLASS,
                    f"request {i} with {ql} query tokens over {kl} keys",
                    "kv_len >= q_len per request (the causal class)",
                )
        if seq_lens is not None:
            given = seq_lens.to("cpu", non_blocking=False).to(torch.int64).tolist()
            if self.kind != "decode" or given != kv_lens:
                raise unsupported(
                    CLASS,
                    "seq_lens",
                    self._PLAN_ARGS_OUTSIDE_THE_CLASS["seq_lens"]
                    + " (on the decode wrapper, seq_lens equal to them)",
                )
        config = engine.default_config(D=D)
        if disable_split_kv:
            from dataclasses import replace

            config = replace(config, split_policy="fused")
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
        if self.graph:
            self._plan_fixed(
                config,
                qo_host,
                kv_indptr_host,
                last_host,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                D,
                page_size,
                int(window_left),
                float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(D),
                int(q_len_per_req),
                non_blocking,
                max_page,
                fast,
                qo_indptr,
                has_sinks,
            )
            return
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
        self._total_q = int(qo_host[-1].item()) if qo_host.numel() else 0
        self._heads = (num_qo_heads, num_kv_heads, D)
        self._page = page_size

    def _plan_fixed(
        self,
        config: engine.AttnConfig,
        qo_host: torch.Tensor,
        kv_indptr_host: torch.Tensor,
        last_host: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        Hq: int,
        Hkv: int,
        D: int,
        page: int,
        window_left: int,
        sm_scale: float,
        q_len_per_req: int,
        non_blocking: bool,
        max_page: int,
        fast: bool,
        qo_indptr: Optional[torch.Tensor] = None,
        has_sinks: Optional[bool] = None,
    ) -> None:
        """The plan of a ``use_cuda_graph=True`` wrapper: the first builds the fixed launch, every one is written into its
        tables and the wrapper's page-table buffers (``engine.FixedLaunch``), after the frozen options are checked. A prefill
        wrapper's kernel knobs (the paired placement, the ring) are those of the first plan; its row capacity is
        ``max_total_num_rows`` or, without it, the first plan's row count."""
        assert self._kv_buffers is not None
        indptr_buf, indices_buf, last_buf = self._kv_buffers
        c = scale_log2(sm_scale)
        frozen = (Hq, Hkv, D, page, window_left, c)
        if self._frozen_plan is not None and frozen != self._frozen_plan:
            names = (
                "num_qo_heads",
                "num_kv_heads",
                "head_dim",
                "page_size",
                "window_left",
                "sm_scale (its binary32 log2 scale)",
            )
            changed = ", ".join(
                f"{n} {new} after {old}"
                for n, new, old in zip(names, frozen, self._frozen_plan, strict=True)
                if new != old
            )
            raise unsupported(
                CLASS,
                f"a plan() of a use_cuda_graph=True wrapper with {changed}",
                "the options of the wrapper's first plan (a captured graph holds them); only the KV lengths and the page "
                "table may change",
            )
        # the sinks declaration selects the captured kernels: the first one given is the wrapper's, and a plan without one
        # (a fast replan has no such argument) keeps it
        if has_sinks is not None:
            if self._has_sinks is not None and bool(has_sinks) != self._has_sinks:
                raise unsupported(
                    CLASS,
                    f"a plan() of a use_cuda_graph=True wrapper with has_sinks={has_sinks} after has_sinks={self._has_sinks}",
                    "the sinks declaration of the wrapper's first plan (a captured graph holds the kernels it selects)",
                )
            if self._frozen_run is not None and self._frozen_run[2] != bool(has_sinks):
                raise unsupported(
                    CLASS,
                    f"has_sinks={has_sinks} on a use_cuda_graph=True wrapper whose first run() was "
                    + ("with" if self._frozen_run[2] else "without")
                    + " sinks",
                    "the presence of sinks of the wrapper's first run()",
                )
        total_q = int(qo_host[-1].item()) if qo_host.numel() else 0
        if self._fixed is None:
            decode = self.kind == "decode"
            if not decode and self._max_rows is None:
                self._max_rows = total_q
            self._fixed = engine.FixedLaunch(
                qo_host.numel() - 1,
                q_len_per_req if decode else None,
                Hq,
                Hkv,
                D,
                page,
                config,
                self.workspace,
                self.device,
                max_rows=None if decode else self._max_rows,
                kv_indptr=indptr_buf,
            )
        else:
            from dataclasses import replace

            # the kernel is the first plan's; the unit kind is this plan's
            config = replace(self._fixed.config, split_policy=config.split_policy)
        planned = engine.plan_attention(
            qo_host,
            kv_indptr_host,
            last_host,
            Hq,
            Hkv,
            D,
            page,
            window_left,
            config,
            fixed=self._fixed,
            non_blocking=non_blocking,
            has_sinks=self._has_sinks if has_sinks is None else bool(has_sinks),
        )
        # the captured kernels read the page table from the wrapper's buffers: refreshed here unless the caller's tensors are
        # those buffers
        for buf, src in (
            (indptr_buf, kv_indptr),
            (indices_buf, kv_indices),
            (last_buf, None if fast else kv_last_page_len),
            (self._qo_buffer, qo_indptr),
        ):
            if buf is None:
                continue
            if src is None or (
                src.device == buf.device and src.data_ptr() == buf.data_ptr()
            ):
                continue
            buf[: src.numel()].copy_(src, non_blocking=non_blocking)
        if has_sinks is not None:
            self._has_sinks = bool(has_sinks)
        self._planned = planned
        self._config = config
        self._kv_indptr = indptr_buf
        self._kv_indices = indices_buf
        self._max_page = max_page
        self._total_kv = int(kv_indptr_host[-1]) if kv_indptr_host.numel() else 0
        self._sm_scale = sm_scale
        self._scale_log2 = c
        self._window_left = window_left
        self._total_q = total_q
        self._heads = (Hq, Hkv, D)
        self._page = page
        self._frozen_plan = frozen

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
        if causal is not None and not causal and self.kind != "decode":
            raise unsupported(
                CLASS, "non-causal attention", "causal attention (bottom-right aligned)"
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
        if q.dim() == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
        if q.dim() != 3 or q.shape[1] != Hq or q.shape[2] != D:
            raise unsupported(
                CLASS, f"q of shape {tuple(q.shape)}", f"(total_q, {Hq}, {D})"
            )
        if self.graph and self.kind == "prefill":
            # the captured launch is built over the caller's q / out of the capture: the plan's rows are its first rows
            if q.shape[0] < self._total_q:
                raise unsupported(
                    CLASS,
                    f"{q.shape[0]} query rows",
                    f"at least the planned {self._total_q}",
                )
        elif q.shape[0] != self._total_q:
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
        q_source = q
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
        lse_nh = None
        if return_lse:
            nh_shape = (q.shape[0], Hq)
            if lse is not None:
                shape = nh_shape if lse_layout == "NH" else nh_shape[::-1]
                bad = engine.check_lse_buffer(
                    lse, q, out, k, v, shape, q_source=q_source
                )
                if bad is not None:
                    raise unsupported(
                        CLASS,
                        f"an lse buffer {bad}",
                        f"a contiguous fp32 {shape} buffer on {self.device} that aliases neither q, out nor the KV cache",
                    )
            # the kernels write the (total_q, Hq) rows; the HN layout is a copy of them
            lse_nh = (
                lse
                if lse is not None and lse_layout == "NH"
                else torch.empty(nh_shape, dtype=torch.float32, device=self.device)
            )
        if self.graph:
            asked = (
                bool(return_lse),
                (lse_base, lse_layout) if return_lse else None,
                sinks is not None,
            )
            if self._frozen_run is None:
                self._frozen_run = asked
            elif asked != self._frozen_run:
                raise unsupported(
                    CLASS,
                    f"run() of a use_cuda_graph=True wrapper with (return_lse, LSE base and layout, sinks given) = {asked} after "
                    f"{self._frozen_run}",
                    "the LSE mode and the presence of sinks of the wrapper's first run() (they select the captured kernels)",
                )
        if sinks is not None:
            # the HN layout's copy writes the caller's buffer after the launch
            written = (out, lse_nh, lse if return_lse else None)
            bad = engine.check_sinks(sinks, Hq, self.device, written)
            if bad is not None:
                raise unsupported(
                    CLASS,
                    f"sinks {bad}",
                    f"a contiguous fp32 ({Hq},) tensor of per-head sink logits on {self.device} that aliases neither out nor lse",
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
