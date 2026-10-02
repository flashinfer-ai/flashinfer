"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import contextlib
import os
from enum import Enum

import torch

from ...push_style_megamoe.shim.protocol import (
    Sm90PushCombine,
    Sm90PushPayload,
    Sm90PushPipe,
    _record_stage,
    _run_guarded_phase,
)
from .cake_gemm import (
    GroupedGemm,
    create_grouped_gemm,
    output_row_capacity,
    validate_grouped_gemm_geometry,
)
from .cake_jit import (
    gen_sm90_cake_bf16_combine_tail_module,
    gen_sm90_cake_bf16_compact_module,
    gen_sm90_cake_bf16_dispatch_module,
)
from .cake_weights import GATE_UP_GROUP, Sm90CakeBf16Weights

__all__ = [
    "FUSED_COMBINE_TAIL_ENV",
    "FUSED_DISPATCH_ENV",
    "FUSED_DISPATCH_MAX_ROUTES",
    "FUSED_DISPATCH_MAX_TOKENS",
    "Sm90CakeBf16MoERunner",
]

# Fast-path kernel selection.  Each toggle defaults to the fused kernel; "0" /
# "false" / "off" / "no" selects the vendored kernels for that phase.  Every
# combination is bit-identical and interoperates with peers on any setting.
#
# Receiver-side combine tail: one kernel (wait for every source, fp32 top-k
# reduce, ack) and no per-round combine-inbox fill, vs. the vendored
# wait_combine -> reduce -> ack kernels with the fill.
FUSED_COMBINE_TAIL_ENV = "FLASHINFER_SM90_CAKE_BF16_FUSED_TAIL"
# Sender-side dispatch: one cooperative launch (count + reserve + store_publish)
# vs. the vendored three kernels.  Used for dedup dispatch with
# T <= FUSED_DISPATCH_MAX_TOKENS and T * top_k <= FUSED_DISPATCH_MAX_ROUTES;
# larger rounds and non-dedup pipes always take the vendored kernels.
FUSED_DISPATCH_ENV = "FLASHINFER_SM90_CAKE_BF16_FUSED_DISPATCH"
FUSED_DISPATCH_MAX_TOKENS = 128
FUSED_DISPATCH_MAX_ROUTES = 1024


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def _fused_combine_tail_default() -> bool:
    return _env_flag(FUSED_COMBINE_TAIL_ENV)


def _fused_dispatch_default() -> bool:
    return _env_flag(FUSED_DISPATCH_ENV)


class _RunnerState(Enum):
    IDLE = "idle"
    STAGED = "staged"
    POISONED = "poisoned"
    DESTROYED = "destroyed"


class Sm90CakeBf16MoERunner:
    """Native BF16 SM90 MegaMoE forward over a :class:`Sm90PushPipe`.

    Round structure (all on the caller's stream):

    ``begin_round`` (round-tag bump, wait for the peers' acks) -> ``dispatch``
    (bf16 payload, P2P push; one cooperative kernel for small rounds, the
    vendored count / reserve / store_publish kernels otherwise) | ``wait_prefix``
    -> ``compact_bf16`` (gather inbox rows expert-major) -> FC1 (Cake WGMMA,
    fused SwiGLU, bf16 intermediate) -> FC2 (Cake WGMMA, bf16 out) ->
    ``combine`` (bf16 wire, fp32 route-weight multiply) -> ``combine_tail`` (one
    kernel: wait for every source, fp32 sum over the unmasked top-k slots, bf16
    output, ack).

    ``fused_combine_tail=False`` (default from ``FUSED_COMBINE_TAIL_ENV``) runs
    the vendored ``wait_combine`` -> ``reduce`` -> ``ack`` kernels with the
    per-round combine-inbox fill instead; ``fused_dispatch=False`` (default
    from ``FUSED_DISPATCH_ENV``) always runs the vendored dispatch kernels.
    Every combination is bit-identical and a peer rank may use any of them.

    Precision contract: bf16 operands into both GEMMs, fp32 accumulation, bf16
    intermediate, bf16 dispatch payload and combine wire, bf16 output.  No FP8
    anywhere.  The pipe must be configured with ``payload_dtype=BF16`` and
    ``combine_dtype=BF16``.
    """

    def __init__(
        self,
        pipe: Sm90PushPipe,
        weights: Sm90CakeBf16Weights,
        *,
        gemm: GroupedGemm | None = None,
        clamp: float | None = None,
        fused_combine_tail: bool | None = None,
        fused_dispatch: bool | None = None,
    ) -> None:
        self.pipe = pipe
        self._state = _RunnerState.IDLE
        self._staged_tokens: int | None = None
        self._staged_topk_ids: torch.Tensor | None = None
        self._staged_stream: torch.cuda.Stream | None = None
        self._staged_stream_capturing = False
        self._caller_ready_event: torch.cuda.Event | None = None
        self._round_event: torch.cuda.Event | None = None
        self._round_stream_id: int | None = None
        self.record_stages = False
        self._clamp = clamp
        self._fused_tail = (
            _fused_combine_tail_default()
            if fused_combine_tail is None
            else bool(fused_combine_tail)
        )
        self._fused_dispatch = (
            _fused_dispatch_default()
            if fused_dispatch is None
            else bool(fused_dispatch)
        )
        self.weights: Sm90CakeBf16Weights | None = None

        def _local_init():
            if pipe.config.payload_dtype != Sm90PushPayload.BF16:
                raise ValueError(
                    "sm90_bf16_bf16_bf16_push_cake requires Sm90PushConfig(payload_dtype=BF16); "
                    f"got {pipe.config.payload_dtype!r}"
                )
            if pipe.config.combine_dtype != Sm90PushCombine.BF16:
                raise ValueError(
                    "sm90_bf16_bf16_bf16_push_cake requires Sm90PushConfig(combine_dtype=BF16); "
                    f"got {pipe.config.combine_dtype!r}"
                )
            if pipe.config.fuse_fc1_epilogue:
                raise ValueError(
                    "sm90_bf16_bf16_bf16_push_cake implements its own fused FC1 epilogue; "
                    "Sm90PushConfig.fuse_fc1_epilogue must be False"
                )
            self._check_weights(weights)
            validate_grouped_gemm_geometry(
                hidden_size=pipe.H,
                intermediate_size=weights.intermediate_size,
                gate_up_group=GATE_UP_GROUP,
                row_capacity=output_row_capacity(pipe.m_cap),
            )
            self.I = weights.intermediate_size
            self.weights = weights
            self._init_buffers()
            return None

        _run_guarded_phase(pipe._comm, pipe.rank, "weights+buffers", _local_init)

        def _jit():
            self.compact_module = gen_sm90_cake_bf16_compact_module().build_and_load()
            self.tail_module = (
                gen_sm90_cake_bf16_combine_tail_module().build_and_load()
                if self._fused_tail
                else None
            )
            self.dispatch_module = (
                gen_sm90_cake_bf16_dispatch_module().build_and_load()
                if self._fused_dispatch
                else None
            )
            self.gemm = gemm if gemm is not None else create_grouped_gemm(clamp=clamp)
            return None

        _run_guarded_phase(pipe._comm, pipe.rank, "cake-bf16-jit", _jit)

    @property
    def fused_combine_tail(self) -> bool:
        """True when the receiver-side combine tail runs as the single fused kernel."""
        return self._fused_tail

    @property
    def fused_dispatch(self) -> bool:
        """True when small dedup rounds dispatch through the single cooperative kernel."""
        return self._fused_dispatch

    def _use_fused_dispatch(self, num_tokens: int) -> bool:
        pipe = self.pipe
        return (
            self._fused_dispatch
            and pipe.config.dedup_dispatch
            and num_tokens <= FUSED_DISPATCH_MAX_TOKENS
            and num_tokens * pipe.K <= FUSED_DISPATCH_MAX_ROUTES
        )

    # ------------------------------------------------------------------ setup
    def _check_weights(self, weights: Sm90CakeBf16Weights) -> None:
        pipe = self.pipe
        if not isinstance(weights, Sm90CakeBf16Weights):
            raise ValueError(
                f"weights must be Sm90CakeBf16Weights (make_sm90_cake_bf16_weights), got {type(weights).__name__}"
            )
        if weights.num_local_experts != pipe.E or weights.hidden_size != pipe.H:
            raise ValueError(
                f"weights describe E={weights.num_local_experts}, H={weights.hidden_size}; "
                f"the pipe has E={pipe.E}, H={pipe.H}"
            )
        if weights.device != pipe.device:
            raise ValueError(f"weights must be on {pipe.device}, got {weights.device}")

    def _init_buffers(self) -> None:
        pipe = self.pipe
        H = pipe.H
        self.m_buf = output_row_capacity(pipe.m_cap)
        dv = pipe.device
        self.a1 = torch.empty(self.m_buf, H, dtype=torch.bfloat16, device=dv)
        self.meta = torch.empty(self.m_buf, 4, dtype=torch.int32, device=dv)
        self.row_expert = torch.empty(self.m_buf, dtype=torch.int32, device=dv)
        self.h2 = torch.empty(self.m_buf, self.I, dtype=torch.bfloat16, device=dv)
        self.y = torch.empty(self.m_buf, H, dtype=torch.bfloat16, device=dv)
        # fused tail: grid-completion counter (the last block resets it to zero)
        self._tail_blocks_done = torch.zeros(1, dtype=torch.int32, device=dv)
        # fused dispatch: per-destination meta / payload bases shared across blocks
        self._dispatch_bases = torch.zeros(2 * pipe.ep, dtype=torch.int32, device=dv)

    def bind_weights(self, weights: Sm90CakeBf16Weights) -> None:
        """Swap the expert weights between rounds (same geometry, same device)."""
        self._require_usable()
        if self._state == _RunnerState.STAGED:
            raise RuntimeError("cannot rebind weights while a round is staged")
        self._check_weights(weights)
        if weights.intermediate_size != self.I:
            raise ValueError(
                f"weights intermediate_size {weights.intermediate_size} does not match the runner's {self.I}"
            )
        self.weights = weights

    # ------------------------------------------------------------- lifecycle
    @property
    def state(self) -> str:
        return self._state.value

    def _require_usable(self) -> None:
        if self._state == _RunnerState.DESTROYED:
            raise RuntimeError("sm90_cake_bf16 runner has been destroyed")
        if self._state == _RunnerState.POISONED:
            raise RuntimeError(
                "sm90_cake_bf16 runner is poisoned by an earlier mid-round failure; "
                "destroy it and build a new layer"
            )

    def _poison(self) -> None:
        if self._state in (_RunnerState.POISONED, _RunnerState.DESTROYED):
            return
        self._state = _RunnerState.POISONED
        self._staged_tokens = None
        self._staged_topk_ids = None
        self._staged_stream = None
        self._staged_stream_capturing = False
        with contextlib.suppress(Exception):
            self.pipe.proto_abort()

    def abort(self) -> None:
        """Poison the active pipe without entering a cross-rank barrier."""
        self._poison()

    def destroy(self) -> None:
        if self._state == _RunnerState.DESTROYED:
            return
        self._state = _RunnerState.DESTROYED
        gemm = getattr(self, "gemm", None)
        if gemm is not None:
            with contextlib.suppress(Exception):
                gemm.destroy()
        self.pipe.destroy()

    # ---------------------------------------------------------------- rounds
    def _validate_round_inputs(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> int:
        pipe = self.pipe
        if x.ndim != 2 or x.shape[1] != pipe.H:
            raise ValueError(f"x must be [num_tokens, {pipe.H}], got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise ValueError(
                f"x must be bf16 (native BF16 dispatch payload), got {x.dtype}"
            )
        num_tokens = int(x.shape[0])
        if num_tokens > pipe.token_capacity:
            raise ValueError(
                f"num_tokens {num_tokens} exceeds token_capacity {pipe.token_capacity}"
            )
        if tuple(topk_ids.shape) != (num_tokens, pipe.K) or tuple(
            topk_weights.shape
        ) != (num_tokens, pipe.K):
            raise ValueError(
                f"topk_ids/topk_weights must be [{num_tokens}, {pipe.K}], got "
                f"{tuple(topk_ids.shape)} / {tuple(topk_weights.shape)}"
            )
        if topk_ids.dtype != torch.int32:
            raise ValueError(f"topk_ids must be int32, got {topk_ids.dtype}")
        if topk_weights.dtype != torch.float32:
            raise ValueError(f"topk_weights must be float32, got {topk_weights.dtype}")
        for name, t in (
            ("x", x),
            ("topk_ids", topk_ids),
            ("topk_weights", topk_weights),
        ):
            if t.device != pipe.device:
                raise ValueError(f"{name} must be on {pipe.device}, got {t.device}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        return num_tokens

    def _validate_output(self, output: torch.Tensor, num_tokens: int) -> None:
        pipe = self.pipe
        if tuple(output.shape) != (num_tokens, pipe.H):
            raise ValueError(
                f"output must be [{num_tokens}, {pipe.H}], got {tuple(output.shape)}"
            )
        if output.dtype != pipe.out_dtype:
            raise ValueError(
                f"output dtype must be {pipe.out_dtype}, got {output.dtype}"
            )
        if output.device != pipe.device or not output.is_contiguous():
            raise ValueError("output must be a contiguous tensor on the pipe's device")

    @staticmethod
    def _current_stream() -> tuple[torch.cuda.Stream, bool]:
        stream = torch.cuda.current_stream()
        return stream, torch.cuda.is_current_stream_capturing()

    def _begin_round(self) -> None:
        """Open the round: round-tag bump and ack wait, in the pipe's order.

        With the fused tail this mirrors ``Sm90PushPipe.proto_begin_round``
        minus the combine-inbox fill: the senders rewrite every unmasked slot of
        a token below ``num_tokens`` each round and the fused tail skips masked
        slots instead of reading zeroed ones, so the fill has no observable
        effect on the output.
        """
        pipe = self.pipe
        if not self._fused_tail:
            pipe.proto_begin_round()
            return
        if pipe._destroyed:
            raise RuntimeError("sm90_push pipe has been destroyed")
        if pipe._poisoned:
            raise RuntimeError(
                "sm90_push pipe is poisoned by an earlier mid-round failure"
            )
        if pipe._round_open:
            raise RuntimeError(
                "sm90_push pipe: previous round was never acked; "
                "the pipe runs ONE round at a time"
            )
        pipe._round_open = True
        pipe.module.sm90_push_bump_tag(pipe._round)
        pipe.module.sm90_push_wait_acks(*pipe._layout_args(), pipe._round)

    def stage_inputs(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> None:
        """Validate one round, then dispatch its bf16 payload and routing tensors."""
        self._require_usable()
        if self._state == _RunnerState.STAGED:
            raise RuntimeError(
                "stage_inputs called twice for one round; call compute first"
            )
        num_tokens = self._validate_round_inputs(x, topk_ids, topk_weights)
        stream, capturing = self._current_stream()
        stream_id = int(stream.cuda_stream)
        if (
            not capturing
            and self._round_event is not None
            and self._round_stream_id != stream_id
            and not self._round_event.query()
        ):
            raise RuntimeError(
                "cannot start a round on a different stream while the previous round is still executing"
            )
        nv = self.record_stages
        try:
            with _record_stage("begin_round", nv):
                self._begin_round()
            with _record_stage("dispatch", nv):
                if self._use_fused_dispatch(num_tokens):
                    self.dispatch_module.sm90_cake_dispatch_fused_bf16(
                        x,
                        topk_ids,
                        topk_weights,
                        *self.pipe._layout_args(),
                        self.pipe._round,
                        self._dispatch_bases,
                    )
                else:
                    self.pipe.proto_dispatch(x, topk_ids, topk_weights)
        except Exception:
            self._poison()
            raise
        self._staged_tokens = num_tokens
        self._staged_topk_ids = topk_ids  # the fused tail skips masked (-1) slots
        self._staged_stream = stream
        self._staged_stream_capturing = capturing
        self._state = _RunnerState.STAGED

    def compute(self, *, output: torch.Tensor) -> torch.Tensor:
        """Finish the staged round and reduce directly into ``output``."""
        self._require_usable()
        if self._state != _RunnerState.STAGED:
            raise RuntimeError("compute requires a preceding stage_inputs call")
        pipe = self.pipe
        weights = self.weights
        num_tokens = self._staged_tokens
        topk_ids = self._staged_topk_ids
        staged_stream = self._staged_stream
        staged_capturing = self._staged_stream_capturing
        assert (
            num_tokens is not None
            and topk_ids is not None
            and staged_stream is not None
            and weights is not None
        )
        try:
            self._validate_output(output, num_tokens)
        except Exception:
            self._poison()
            raise
        staged_stream_id = int(staged_stream.cuda_stream)
        caller_stream, _caller_capturing = self._current_stream()
        streams_differ = int(caller_stream.cuda_stream) != staged_stream_id
        if streams_differ:
            if self._caller_ready_event is None:
                self._caller_ready_event = torch.cuda.Event()
            self._caller_ready_event.record(caller_stream)
            staged_stream.wait_event(self._caller_ready_event)
        stream_context = (
            contextlib.nullcontext()
            if not streams_differ
            else torch.cuda.stream(staged_stream)
        )
        nv = self.record_stages
        try:
            with stream_context:
                with _record_stage("wait_prefix", nv):
                    pipe.proto_wait_prefix()
                with _record_stage("compact_bf16", nv):
                    self.compact_module.sm90_cake_compact_bf16(
                        self.a1,
                        self.meta,
                        self.row_expert,
                        *pipe._layout_args(),
                        pipe._seg_src_base,
                        pipe._seg_out_base,
                        pipe._m_dev,
                        pipe._next_row,
                    )
                with _record_stage("fc1", nv):
                    self.gemm.fc1(self.a1, weights.w13, pipe._offsets, self.h2)
                with _record_stage("fc2", nv):
                    self.gemm.fc2(self.h2, weights.w2, pipe._offsets, self.y)
                with _record_stage("combine", nv):
                    pipe.proto_combine(self.y, self.meta)
                if self._fused_tail:
                    with _record_stage("combine_tail", nv):
                        self.tail_module.sm90_cake_combine_tail_bf16(
                            output,
                            topk_ids,
                            *pipe._layout_args(),
                            pipe._round,
                            pipe._lc,
                            pipe._done,
                            self._tail_blocks_done,
                            num_tokens,
                        )
                    pipe._round_open = False  # the fused tail performed the ack
                else:
                    with _record_stage("wait_combine", nv):
                        pipe.proto_wait_combine()
                    with _record_stage("reduce", nv):
                        pipe.proto_reduce(output, num_tokens)
                    with _record_stage("ack", nv):
                        pipe.proto_ack()
        except Exception:
            self._poison()
            raise
        self._state = _RunnerState.IDLE
        self._staged_tokens = None
        self._staged_topk_ids = None
        self._staged_stream = None
        self._staged_stream_capturing = False
        if not staged_capturing or streams_differ:
            if self._round_event is None:
                self._round_event = torch.cuda.Event()
            self._round_event.record(staged_stream)
            self._round_stream_id = staged_stream_id
            if streams_differ:
                caller_stream.wait_event(self._round_event)
        return output

    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Submit a complete round."""
        num_tokens = self._validate_round_inputs(x, topk_ids, topk_weights)
        self._validate_output(output, num_tokens)
        self.stage_inputs(x, topk_ids, topk_weights)
        return self.compute(output=output)
