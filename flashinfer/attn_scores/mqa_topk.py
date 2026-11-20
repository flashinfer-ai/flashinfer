# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public entry point for continuous MQA TopK."""

from __future__ import annotations

from typing import Any, Literal

import torch

from ..api_logging import flashinfer_experimental_api, flashinfer_api


class FP8MQATopKWrapper:
    """Experimental selective TopK on continuous K for prefill.

    Every wrapper shares a process-wide retirement queue. ``plan``, ``prepare_pool`` and
    ``clear_pools`` drain it unless their own stream is capturing. Do not call
    these methods while any thread is performing global capture on another
    stream. Retired modules remain loaded until the next drain or process exit.

    FP8 uses K[N,D] and FP32 scales[N];
    MXFP4 uses packed uint8 K[N,D/2] and UE8M0 uint8 scales[N,4];
    MXFP8 uses FP8 K[N,D] with the same block-scale layout.
    ``num_keys`` describes storage, while plan's prefixes describe live tokens
    before compression. It defaults to floor(max_context_length / ratio).
    MXFP4 and MXFP8 storage must be padded to a multiple of 256 rows.

    Selective scoring infers its model policy from ``num_heads``, ``head_dim``,
    ``top_k`` and ``compression_ratio``. An optional ``model_config`` name
    must agree with those fields. Explicit Qwen configuration requires weights ``1/sqrt(128)``
    rounded to the selected weights dtype and, for FP8, unit K scales;
    these conditions are checked on replay.
    Length policies use physical K rows, including the queries' compressed keys.

    ``weights_dtype`` defaults to ``torch.float32``. Selective scoring also accepts
    ``torch.bfloat16`` for every supported QK, logits and arithmetic mode.
    Weights must match the planned dtype; BF16 weights may have padded rows
    with contiguous heads, including H4 row stride of eight.
    Input storage dtype does not select head accumulation precision.

    Selective ``math_mode=None`` resolves to ``relu_fast`` for FP8/MXFP4 and
    ``relu_strict`` for MXFP8. Fast mode
    checks bounds on device every replay: weights must be finite and zero or
    have magnitude at least 2**-125, and MXFP4 scale bytes must be <= 159.
    Violations set an error bit and return -1 rows.

    ``math_mode="native_bf16"`` and ``math_mode="native_fp16"`` are opt-in
    modes for H4/H32/H64 FP8, MXFP8 or MXFP4 inputs, requiring the
    matching logits dtype.
    QK dots accumulate in FP32. ReLU results and weights are rounded to the
    native dtype, followed by four native FMA chains and the rounded tree
    ``(c0+c2)+(c1+c3)``. FP8 native_fp16 first divides QK by 512 and multiplies
    weights by 512. This power-of-two rebalance follows from the E4M3FN/D128
    maximum dot product; it avoids overflowing the FP16 conversion of raw QK.
    It is independent of input values. Other native modes use no rebalance.
    FP8 applies the original FP32 key scale after the sum, then rounds the
    output. MXFP4 block scales are applied within the dot product.
    Native modes have their own rounding, underflow and overflow behavior;
    they differ from FP32 head accumulation in ``relu_strict``/``relu_fast``.

    ``backend="auto"`` routes between the named backends and preserves this
    common contract: the "specialized" backend's narrower numerical
    contract (unit ``k_scale``, positive per-row-constant ``weights``) is
    verified on every eager call, and a violating eager call falls back to
    "cutedsl" (when the configuration permits) instead of raising. Because
    those value checks cannot make host decisions inside a captured graph,
    auto REFUSES to capture a specialized run unless the wrapper was
    constructed with ``declare_specialized_contract=True`` — the caller's
    documented assertion that every graph's ``weights``/``k_scale`` satisfy
    the specialized contract for the graphs' lifetime. Declared captures
    still record device-side checks: a replay whose live values break the
    declaration returns all ``-1`` rows and sets ``repair_error`` bit 16
    rather than a selection computed from stale values.

    For single-request capacity (max_batch_size=1), auto prefers specialized
    only for full-prefill live lengths up to 4096. This is a bounded
    completed-call performance policy measured on GB300/CUDA 13.0, not a
    universal latency guarantee. Other lengths use CuTe when it can serve the
    configuration; that resolution stays CuTe on later replans. Explicit
    specialized and multi-request capacities
    retain their existing supported domains.

    Construct outside capture, plan once per step, and run once per layer.
    Prepare every K/scale binding before capture. Keep this wrapper and all
    capture inputs alive until the caller discards its graphs. Calls on a
    wrapper must be serialized. Output is shared int32[Q, top_k] storage with
    request-local indices and -1 padding; ties need not have stable indices.
    """

    is_experimental = True

    @flashinfer_experimental_api(feature="FP8MQATopKWrapper")
    def __init__(
        self,
        *,
        max_context_length: int,
        max_chunk_size: int,
        max_batch_size: int,
        device,
        strategy: Literal["selective"] = "selective",
        backend: Literal["auto", "cutedsl", "specialized"] = "cutedsl",
        num_heads: int,
        head_dim=128,
        top_k=512,
        math_mode=None,
        compression_ratio=1,
        num_keys=None,
        qkv_dtype="fp8",
        logits_dtype=torch.float32,
        weights_dtype=torch.float32,
        model_config=None,
        declare_specialized_contract=False,
    ):
        if strategy != "selective":
            raise ValueError("continuous TopK supports selective only")
        if backend not in ("auto", "cutedsl", "specialized"):
            raise ValueError("backend must be 'auto', 'cutedsl' or 'specialized'")
        if declare_specialized_contract and backend == "cutedsl":
            raise ValueError(
                "declare_specialized_contract applies to backend='auto' "
                "(and is implicit for backend='specialized')"
            )

        options = dict(
            max_context_length=max_context_length,
            max_chunk_size=max_chunk_size,
            max_batch_size=max_batch_size,
            device=device,
            num_heads=num_heads,
            head_dim=head_dim,
            top_k=top_k,
            compression_ratio=compression_ratio,
            math_mode=math_mode,
            num_keys=num_keys,
        )
        options.update(
            qkv_dtype=qkv_dtype,
            logits_dtype=logits_dtype,
            weights_dtype=weights_dtype,
            model_config=model_config,
        )
        self._capacity_plan: Any
        if backend == "specialized":
            # Agent-generated SM103 kernel; adds multi-request support
            # (qo/kv_indptr of shape [B + 1]). Inherits the functional
            # entry's validated-toolkit refusal semantics. See
            # flashinfer/attn_scores/mqa_topk_specialized.py for the
            # validated contract and restrictions.
            from .mqa_topk_specialized import SpecializedTopKPlan

            self._capacity_plan = SpecializedTopKPlan(**options)
        elif backend == "auto":
            # "auto" is purely a router over the named backends (cutedsl,
            # specialized): it prefers "specialized" when the static
            # configuration and toolkit allow, demotes to "cutedsl" on the
            # first rejected prepare/plan (replaying prepared bindings),
            # falls back to "cutedsl" when an eager run supplies values
            # outside the specialized numerical contract, and refuses to
            # CAPTURE the specialized backend unless the caller declared
            # that contract (see _AutoTopKPlan).
            self._capacity_plan = _AutoTopKPlan(
                options, declared_contract=declare_specialized_contract
            )
        else:
            from ..experimental.attn_scores import _continuous_plan_type

            self._capacity_plan = _continuous_plan_type("cutedsl")(**options)
        self._compression_ratio = compression_ratio
        self._backend = backend
        self._strategy = strategy

    @property
    def resolved_backend(self):
        """The named backend serving this wrapper ("auto" resolves lazily)."""
        if self._backend != "auto":
            return self._backend
        return self._capacity_plan.resolved_backend

    @flashinfer_api
    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        *,
        num_query_rows: int | None = None,
        next_n: int | None = None,
        host_length_hints: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Stage original-token prefixes for the selected backend.

        Specialized (including auto resolved to specialized) takes a pair of
        CUDA int32 ``[B+1]`` prefixes for full prefill, with equal Q and KV
        lengths per request and 256-token-aligned KV starts. Plan eagerly;
        ``host_length_hints`` is unsupported. Discard graphs using the old
        specialized plan before replanning: replacing its metadata and scratch
        invalidates addresses recorded by those graphs. Capture a new run
        after eager preparation of the new plan.

        The single-request prefix, error-bit and capturable-plan details below
        describe cutedsl (including auto resolved to cutedsl).

        For query i, visible keys end at
        floor((KV - Q + i + 1) / compression_ratio).
        K stores floor(KV / compression_ratio) keys.
        ``next_n`` optionally asserts exactly that many query rows.
        ``host_length_hints`` optionally supplies host upper bounds; violations
        set error bit 2. Invalid prefixes fail closed (bits 1/2/4). Read
        ``error`` and ``repair_error`` outside capture; the latter sets bit 16
        on fast-mode or explicit Qwen numerical contract violations. Planning
        is capturable after eager preparation.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            CUDA int32 query offsets in original tokens: ``[2]`` for cutedsl,
            ``[B+1]`` for specialized.
        kv_indptr : torch.Tensor
            CUDA int32 KV offsets in original tokens, same shape as ``qo_indptr``.
        num_query_rows : int, optional
            Extent of the supplied Q storage. Defaults to ``max_chunk_size``.
        next_n : int, optional
            Assert that the request has exactly this many query rows.
        host_length_hints : tuple[torch.Tensor, torch.Tensor], optional
            ``(query_prefix, kv_bounds)`` as contiguous CPU int32 or int64
            tensors of shapes ``[2]`` and ``[1]``. The query prefix must fit
            the supplied Q extent, and each KV bound must lie in
            ``[0, max_context_length]``. Device violations set error bit 2.
            Omit when capturing ``plan`` in a CUDA graph.

        Returns
        -------
        None
        """
        return self._capacity_plan.plan(
            qo_indptr,
            kv_indptr,
            num_query_rows=num_query_rows,
            next_n=next_n,
            host_length_hints=host_length_hints,
        )

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        *,
        k_scale: torch.Tensor,
        out: torch.Tensor | None = None,
        q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score continuous K directly; no KV packing, page lookup or gather.

        Parameters
        ----------
        q : torch.Tensor
            Contiguous CUDA queries. FP8 and MXFP8 use E4M3FN ``[Q, H, D]``;
            MXFP4 uses packed uint8 ``[Q, H, D/2]``.
        weights : torch.Tensor
            CUDA head weights ``[Q, H]`` with the configured FP32 or BF16 dtype.
            FP32 weights are contiguous; BF16 rows may be padded.
        k : torch.Tensor
            Contiguous CUDA keys: E4M3FN ``[N, D]`` for FP8 and MXFP8, or
            packed uint8 ``[N, D/2]`` for MXFP4. Block-scaled storage has
            ``N`` divisible by 256; the plan specifies the live extent.
        k_scale : torch.Tensor
            Contiguous CUDA FP32 ``[N]`` scales for FP8, or UE8M0 uint8
            ``[N, 4]`` block scales for MXFP4 and MXFP8. K and its scales
            must be 16-byte aligned and prepared before capture.
        out : torch.Tensor, optional
            Contiguous CUDA int32 output buffer ``[Q, top_k]``.
        q_scale : torch.Tensor, optional
            Required for MXFP4 and MXFP8: contiguous CUDA int32 ``[Q, H]``
            or UE8M0 uint8 ``[Q, H, 4]`` block scales. Must be None for FP8.

        Returns
        -------
        torch.Tensor
            CUDA int32 ``[Q, top_k]`` request-local indices, padded with -1.
            Storage is shared across calls unless ``out`` is supplied.
        """
        return self._capacity_plan.run(
            q, weights, k, k_scale=k_scale, out=out, q_scale=q_scale
        )

    def prepare_pool(self, k, *, k_scale):
        """Compile/capture a K/scale binding outside caller graph capture."""
        return self._capacity_plan.prepare_pool(k, k_scale=k_scale)

    def clear_pools(self):
        return self._capacity_plan.clear_pools()

    def input_buffers(self):
        """Stable Q/weight/prefix buffers (plus q_scale for MXFP4/MXFP8)."""
        return self._capacity_plan.input_buffers()

    @property
    def strategy(self):
        return self._strategy

    @property
    def math_mode(self):
        return self._capacity_plan.math_mode

    @property
    def compression_ratio(self):
        return self._compression_ratio

    @property
    def error(self):
        return self._capacity_plan.error

    @property
    def repair_error(self):
        return self._capacity_plan.repair_error

    @property
    def active_shape(self):
        return self._capacity_plan.active_shape


class _AutoTopKPlan:
    """Router plan for ``backend="auto"``: prefers the "specialized"
    backend when the configuration, toolkit and planned workload allow, and
    demotes to "cutedsl" on the first rejected ``prepare_pool``/``plan``
    (replaying already-prepared bindings). "auto" carries no behavior of
    its own — every call is served by one of the named backends.

    Admission facts for the specialized resolution, which is narrower than
    the common wrapper contract (unit ``k_scale``, positive per-row-constant
    ``weights``) in values only ``run`` supplies:

    - ``prepare_pool`` verifies each binding's ``k_scale`` is all-ones at
      prepare time (a non-unit binding demotes or falls back to cutedsl);
    - every EAGER ``run`` re-verifies both value conditions, and a
      violation falls back to cutedsl (``_demote_active``) instead of
      raising — the common contract serves those values;
    - a CAPTURED ``run`` cannot fall back (cutedsl prepare/plan are not
      capturable), so capture of the specialized backend is REFUSED unless
      the caller declared the contract for the graphs' lifetime
      (``declare_specialized_contract=True``). Declared captures rely on
      the specialized plan's device-side replay checks, which fail closed
      (-1 rows, ``repair_error`` bit 16) on violation.

    While the specialized backend is active the router keeps the binding
    log and the latest plan arguments so a run-time fallback can rebuild
    cutedsl state; a retired specialized plan is kept alive because
    captured graphs may hold raw addresses into its plan-owned tensors.
    Once cutedsl is active the resolution is terminal and the logs are
    dropped."""

    def __init__(self, options, declared_contract=False):
        self._options = dict(options)
        self._declared = bool(declared_contract)
        self._bindings = []
        self._last_plan = None
        self._retired = []
        self._active = None
        self._cand = None
        self._demote_reason = None
        try:
            from .mqa_topk_specialized import SpecializedTopKPlan

            self._cand = SpecializedTopKPlan(**self._options)
            self._cand_name = "specialized"
        except ValueError as e:
            self._demote_reason = str(e)
            try:
                self._cand = self._make_cutedsl()
            except ValueError as e2:
                raise ValueError(
                    "auto: no backend can serve this configuration "
                    f"(specialized: {e}; cutedsl: {e2})"
                ) from e2
            self._cand_name = "cutedsl"

    def _make_cutedsl(self):
        from ..experimental.attn_scores import _continuous_plan_type

        return _continuous_plan_type("cutedsl")(**self._options)

    @property
    def resolved_backend(self):
        return self._cand_name if self._active is None else self._active_name

    def _demote(self, reason):
        if self._cand_name != "specialized":
            raise RuntimeError(
                "auto: cutedsl candidate failed and no further backend "
                f"exists ({reason}; earlier: {self._demote_reason})"
            )
        # Build and fully prepare the replacement before publishing it: a
        # transient failure while replaying bindings must leave the router
        # retryable (candidate unchanged), not half-demoted with a binding
        # missing from the new candidate.
        cand = self._make_cutedsl()
        for k, k_scale in self._bindings:
            cand.prepare_pool(k, k_scale=k_scale)
        self._demote_reason = reason
        self._cand = cand
        self._cand_name = "cutedsl"

    def _demote_active(self, reason):
        """Run-time fallback: the active specialized backend rejected
        VALUES the common wrapper contract serves (ContractViolation).
        Rebuild cutedsl, replay the recorded bindings and the latest plan,
        and retire the specialized plan without freeing it (captured
        graphs may hold raw addresses into its plan-owned tensors)."""
        try:
            cand = self._make_cutedsl()
        except ValueError as e:
            raise ValueError(
                "auto: the specialized backend rejected runtime values "
                f"outside its contract ({reason}) and cutedsl cannot serve "
                f"this configuration ({e})"
            ) from e
        for bk, bk_scale in self._bindings:
            cand.prepare_pool(bk, k_scale=bk_scale)
        if self._last_plan is not None:
            qo, kv, kw = self._last_plan
            cand.plan(qo, kv, **kw)
        self._retired.append(self._active)
        self._demote_reason = reason
        self._cand = cand
        self._active = cand
        self._cand_name = "cutedsl"
        self._active_name = "cutedsl"
        self._bindings = []
        self._last_plan = None

    def prepare_pool(self, k, *, k_scale):
        if self._active is not None:
            if self._active_name != "specialized":
                return self._active.prepare_pool(k, k_scale=k_scale)
            from .mqa_topk_specialized import ContractViolation

            try:
                out = self._active.prepare_pool(k, k_scale=k_scale)
            except ContractViolation as e:
                self._demote_active(str(e))
                return self._active.prepare_pool(k, k_scale=k_scale)
            self._bindings.append((k, k_scale))
            return out
        try:
            out = self._cand.prepare_pool(k, k_scale=k_scale)
        except ValueError as e:
            self._demote(str(e))
            # _demote replayed only the earlier bindings; the binding that
            # was rejected must be prepared on the new candidate too, or a
            # caller who prepared every binding before capture would hit an
            # unprepared-binding raise from cutedsl under capture.
            out = self._cand.prepare_pool(k, k_scale=k_scale)
        self._bindings.append((k, k_scale))
        return out

    def plan(self, qo_indptr, kv_indptr, **kw):
        cand = self._cand if self._active is None else self._active
        name = self._cand_name if self._active is None else self._active_name
        reason = None
        performance_miss = False
        if name == "specialized":
            try:
                state = cand._prepare_plan(qo_indptr, kv_indptr, **kw)
            except ValueError as e:
                reason = str(e)
            else:
                # Completed public-wrapper measurements on GB300 qualify this
                # bounded B1 region. Use validated live prefixes, not capacity.
                # CuTe cannot serve max_batch_size > 1, even with one live
                # request, so those configurations retain specialized support.
                if self._options["max_batch_size"] == 1 and state["T"] > 4096:
                    reason = "single-request length outside measured auto region"
                    performance_miss = True
                else:
                    cand._commit_plan(state)
            if reason is not None:
                try:
                    replacement = self._make_cutedsl()
                    for k, k_scale in self._bindings:
                        replacement.prepare_pool(k, k_scale=k_scale)
                    out = replacement.plan(qo_indptr, kv_indptr, **kw)
                except ValueError:
                    if not performance_miss:
                        raise
                    # Performance preference must not remove configurations
                    # that only specialized can serve (e.g. unaligned capacity).
                    cand._commit_plan(state)
                    reason, out = None, None
                else:
                    # Publish only after bindings and the incoming plan succeed.
                    # A failed fallback leaves the prior specialized state usable.
                    cand, name = replacement, "cutedsl"
            else:
                out = None
        else:
            out = cand.plan(qo_indptr, kv_indptr, **kw)
        self._cand, self._cand_name = cand, name
        self._active, self._active_name = cand, name
        if reason is not None:
            self._demote_reason = reason
        if name == "specialized":
            self._last_plan = (qo_indptr, kv_indptr, dict(kw))
        else:
            # CuTe is terminal, including later replans back into the region.
            self._bindings = []
            self._last_plan = None
        return out

    def _resolved(self):
        if self._active is None:
            raise RuntimeError("auto: plan must run before this call")
        return self._active

    def run(self, q, weights, k, *, k_scale, out=None, q_scale=None):
        plan_ = self._resolved()
        if self._active_name != "specialized":
            return plan_.run(q, weights, k, k_scale=k_scale, out=out, q_scale=q_scale)
        if not self._declared and plan_._is_capturing():
            raise RuntimeError(
                "auto: refusing to capture the specialized backend. Its "
                "kernel reads neither weights nor k_scale, and a captured "
                "graph cannot fall back to cutedsl, so runtime updates to "
                "those tensors could not change the replayed selection. "
                "Construct the wrapper with declare_specialized_contract="
                "True to assert unit k_scale and positive per-row-constant "
                "weights for the graphs' lifetime, or use backend='cutedsl' "
                "for captured runs."
            )
        from .mqa_topk_specialized import ContractViolation

        try:
            return plan_.run(q, weights, k, k_scale=k_scale, out=out, q_scale=q_scale)
        except ContractViolation as e:
            # an eager call with values the common contract serves: fall
            # back to cutedsl and serve the call (ContractViolation is only
            # raised outside capture)
            self._demote_active(str(e))
            return self._active.run(
                q, weights, k, k_scale=k_scale, out=out, q_scale=q_scale
            )

    def clear_pools(self):
        backend = self._cand if self._active is None else self._active
        out = backend.clear_pools()
        # cleared bindings must neither be replayed by a later demotion
        # nor kept alive by the log after the backend dropped them
        self._bindings = []
        return out

    def input_buffers(self):
        if self._active is None and self._cand_name == "cutedsl":
            # the candidate is already the terminal backend (selected at
            # construction or by an earlier demotion): its stable buffers
            # exist before the first plan and no further demotion can
            # invalidate them, so expose them for staging the first plan
            return self._cand.input_buffers()
        return self._resolved().input_buffers()

    @property
    def math_mode(self):
        return (self._active or self._cand).math_mode

    @property
    def error(self):
        return (self._active or self._cand).error

    @property
    def repair_error(self):
        return (self._active or self._cand).repair_error

    @property
    def active_shape(self):
        return (self._active or self._cand).active_shape


__all__ = ["FP8MQATopKWrapper"]
