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
import shutil
import socket
from enum import Enum

import torch

from .protocol import (
    Sm90PushPipe,
    _record_stage,
    _run_guarded_phase,
)
from .weights import Sm90PushWeights

__all__ = ["Sm90PushMoERunner"]


class _RunnerState(Enum):
    IDLE = "idle"
    STAGED = "staged"
    POISONED = "poisoned"
    DESTROYED = "destroyed"


class Sm90PushMoERunner:
    """Full SM90 push MegaMoE forward over a :class:`Sm90PushPipe`."""

    def __init__(
        self,
        pipe: Sm90PushPipe,
        weights: "Sm90PushWeights | torch.Tensor | None" = None,
        w13_sf: torch.Tensor | None = None,
        w2_fp8: torch.Tensor | None = None,
        w2_sf: torch.Tensor | None = None,
        *,
        w13_fp8: torch.Tensor | None = None,
        w13_interleaved: bool | None = None,
    ):
        """Full forward executor; weights via Sm90PushWeights bundle or raw tensors."""
        E, H = pipe.E, pipe.H
        self.pipe = pipe
        self._state = _RunnerState.IDLE
        self._staged_tokens: int | None = None
        self._staged_stream: torch.cuda.Stream | None = None
        self._staged_stream_capturing = False
        self._staged_output: torch.Tensor | None = None
        self._caller_ready_event: torch.cuda.Event | None = None
        self._round_event: torch.cuda.Event | None = None
        self._round_stream_id: int | None = None
        self.record_stages = False  # per-stage profiler ranges (see _record_stage)

        # weights are per-rank state: form AND content checks run guarded
        def _local_init():
            nonlocal weights, w13_sf, w2_fp8, w2_sf, w13_fp8, w13_interleaved
            if w13_fp8 is not None:  # legacy keyword form: w13_fp8= names arg 1
                if weights is not None:
                    raise ValueError(
                        "pass the weights either positionally/as a bundle OR "
                        "via the legacy w13_fp8= keyword, not both"
                    )
                weights = w13_fp8
            if weights is None:
                raise ValueError(
                    "weights are required: pass an Sm90PushWeights bundle or "
                    "the four raw tensors (w13_fp8, w13_sf, w2_fp8, w2_sf)"
                )
            if isinstance(weights, Sm90PushWeights):
                if w13_sf is not None or w2_fp8 is not None or w2_sf is not None:
                    raise ValueError(
                        "pass EITHER an Sm90PushWeights bundle OR the four raw "
                        "tensors, not both"
                    )
                if (
                    w13_interleaved is not None
                    and w13_interleaved != weights.w13_interleaved
                ):
                    raise ValueError(
                        f"explicit w13_interleaved={w13_interleaved} contradicts "
                        f"the Sm90PushWeights tag ({weights.w13_interleaved}); "
                        f"drop the kwarg -- the bundle already knows its layout"
                    )
                w13_fp8, w13_sf = weights.w13_fp8, weights.w13_sf
                w2_fp8, w2_sf = weights.w2_fp8, weights.w2_sf
                w13_interleaved = weights.w13_interleaved
            else:
                w13_fp8 = weights
                if w13_sf is None or w2_fp8 is None or w2_sf is None:
                    raise ValueError(
                        "raw-tensor form requires all four tensors "
                        "(w13_fp8, w13_sf, w2_fp8, w2_sf)"
                    )
                if w13_interleaved is None:
                    w13_interleaved = False
            if pipe.config.fuse_fc1_epilogue != w13_interleaved:
                raise ValueError(
                    f"fuse_fc1_epilogue={pipe.config.fuse_fc1_epilogue} requires "
                    f"w13_interleaved={pipe.config.fuse_fc1_epilogue} (transform "
                    f"with interleave_gate_up={pipe.config.fuse_fc1_epilogue}); "
                    f"got w13_interleaved={w13_interleaved}"
                )
            if (
                w13_fp8.dtype != torch.float8_e4m3fn
                or w2_fp8.dtype != torch.float8_e4m3fn
            ):
                raise ValueError(
                    "weights must be pre-quantized fp8 "
                    "(use transform_weights_for_sm90_push)"
                )
            if w13_fp8.ndim != 3 or w13_fp8.shape[0] != E or w13_fp8.shape[2] != H:
                raise ValueError(f"w13_fp8 must be (E={E}, 2I, H={H})")
            two_i = w13_fp8.shape[1]
            if two_i <= 0 or two_i % 256 != 0:
                raise ValueError(
                    f"w13 second dim (2I) must be a positive multiple of 256 "
                    f"(I % 128 == 0), got {two_i}"
                )
            if w13_sf.dtype != torch.float32 or w2_sf.dtype != torch.float32:
                raise ValueError(
                    f"weight scales must be float32 (the kernels reinterpret "
                    f"the buffers as float*), got w13_sf={w13_sf.dtype}, "
                    f"w2_sf={w2_sf.dtype}"
                )
            i_size = two_i // 2
            if tuple(w2_fp8.shape) != (E, H, i_size):
                raise ValueError(f"w2_fp8 must be (E, H, I) = ({E}, {H}, {i_size})")
            if tuple(w13_sf.shape) != (E, two_i // 128, H // 128) or tuple(
                w2_sf.shape
            ) != (E, H // 128, i_size // 128):
                raise ValueError("weight scale shapes do not match the fp8 weights")
            for name, t in (
                ("w13_fp8", w13_fp8),
                ("w13_sf", w13_sf),
                ("w2_fp8", w2_fp8),
                ("w2_sf", w2_sf),
            ):
                if t.device != pipe.device:
                    raise ValueError(f"{name} must be on {pipe.device}, got {t.device}")
                if not t.is_contiguous():
                    raise ValueError(f"{name} must be contiguous")
            self.w13_interleaved = w13_interleaved
            self.I = i_size
            self.w13_fp8, self.w13_sf = w13_fp8, w13_sf
            self.w2_fp8, self.w2_sf = w2_fp8, w2_sf
            self._init_gemm_resources()
            return None

        _run_guarded_phase(
            pipe._comm, getattr(pipe, "rank", 0), "weights+gemm-resources", _local_init
        )
        self._prepare_gemm_jit_collective()

    def _init_gemm_resources(self) -> None:
        """Local (collective-free) resource construction; see __init__."""
        from .gemm import create_sm90_push_fp8_moe_gemm_runner

        pipe = self.pipe
        E, H = pipe.E, pipe.H
        two_i = 2 * self.I
        self.runner = create_sm90_push_fp8_moe_gemm_runner()

        m_cap = pipe.m_cap
        m_buf = (m_cap + 127) // 128 * 128
        dv = pipe.device
        p_ws = max((m_cap + E * 31) // 32 * 32, 1)
        self.a1 = torch.empty(m_buf, H, dtype=torch.uint8, device=dv)
        self.sfa1 = torch.empty((H // 128) * p_ws + 128, dtype=torch.float32, device=dv)
        self.meta = torch.empty(m_buf, 4, dtype=torch.int32, device=dv)
        self.row_expert = torch.empty(m_buf, dtype=torch.int32, device=dv)
        self.h = (
            None
            if pipe.config.fuse_fc1_epilogue
            else torch.empty(m_buf, two_i, dtype=torch.bfloat16, device=dv)
        )
        self.a2 = torch.empty(m_buf, self.I, dtype=torch.uint8, device=dv)
        self.sfa2 = torch.empty(
            (self.I // 128) * p_ws + 128, dtype=torch.float32, device=dv
        )
        self.y = torch.empty(m_buf, H, dtype=torch.bfloat16, device=dv)
        self._g = None  # lazy: only the unfused (fuse_act=False) path needs it

        self._workspace: torch.Tensor | None = None
        self.configure_workspace()

    def configure_workspace(self) -> None:
        """(Re)apply this pipeline's workspace state to its runner."""
        pipe = self.pipe
        two_i = 2 * self.I
        sz = self.runner.get_moe_workspace_size(
            pipe.token_capacity * pipe.K,
            pipe.m_cap,
            max(two_i, pipe.H),
            max(pipe.H, self.I),
            pipe.E,
            True,
            True,
        )
        self._workspace = torch.empty(
            max(int(sz), 1), device=pipe.device, dtype=torch.uint8
        )
        self.runner.configure_workspace(self._workspace)

    _FC1_FUSED_FAIL_HELP = (
        "moe_gemm_fc1_fused failed (the private fused FP8 variant is "
        "loaded from its disk cache or JIT-compiled via in-process nvcc). "
        "Likely causes and fixes: "
        "(1) stale/corrupt DeepGEMM disk cache -- clear "
        "~/.tensorrt_llm/cache (or $TRTLLM_DG_CACHE_DIR) and retry; "
        "(2) nvcc not reachable (CUDA_HOME) or DeepGEMM JIT disabled "
        "(TRTLLM_DG_ENABLED=0). To run without this private fused path, rebuild "
        "with Sm90PushConfig(fuse_fc1_epilogue=False) and non-interleaved "
        "weights (interleave_gate_up=False) -- the unfused FA path is the "
        "maintained anchor."
    )

    @staticmethod
    def _nvcc_available(command: str) -> bool:
        expanded = os.path.expandvars(os.path.expanduser(command))
        return shutil.which(expanded) is not None

    def _missing_gemm_jit_caches(self) -> list[str]:
        pipe = self.pipe
        missing = []
        if pipe.config.fuse_fc1_epilogue and not bool(
            self.runner.is_moe_gemm_fc1_fused_jit_cache_ready(2 * self.I, pipe.H)
        ):
            missing.append("fused-fc1")
        if not pipe.config.fuse_fc1_epilogue and not bool(
            self.runner.is_moe_gemm_jit_cache_ready(2 * self.I, pipe.H)
        ):
            missing.append("fc1")
        if not bool(self.runner.is_moe_gemm_jit_cache_ready(pipe.H, self.I)):
            missing.append("fc2")
        return missing

    def _require_gemm_jit_runtime(self) -> None:
        pipe = self.pipe
        if pipe.config.fuse_fc1_epilogue and not bool(
            self.runner.is_deepgemm_jit_enabled()
        ):
            raise RuntimeError(
                "fuse_fc1_epilogue=True requires DeepGEMM JIT; "
                "TRTLLM_DG_ENABLED=0 or missing JIT include directories only "
                "provide the unfused CUTLASS fallback"
            )

        missing = self._missing_gemm_jit_caches()
        compiler = str(self.runner.get_deepgemm_nvcc_compiler())
        if missing and not self._nvcc_available(compiler):
            cache_dir = str(self.runner.get_deepgemm_cache_dir())
            fallback = (
                " Set TRTLLM_DG_ENABLED=0 before loading the module to use the "
                "fixed-tactic unfused CUTLASS fallback."
                if not pipe.config.fuse_fc1_epilogue
                else " Prewarm compatible fused-FC1 and FC2 cubins on this GPU shape "
                "or install NVCC."
            )
            raise RuntimeError(
                "SM90 push FP8 has cold DeepGEMM cache miss(es) for "
                f"{', '.join(missing)} in {cache_dir}, but NVCC compiler "
                f"{compiler!r} is not executable.{fallback}"
            )

    def _gemm_jit_cache_identity(
        self,
    ) -> tuple[str, str, int, int, int, bool, int, int, int, int, bool, bool]:
        pipe = self.pipe
        props = torch.cuda.get_device_properties(pipe.device)
        cache_dir = os.path.normcase(
            os.path.realpath(str(self.runner.get_deepgemm_cache_dir()))
        )
        compiler = str(self.runner.get_deepgemm_nvcc_compiler())
        return (
            socket.gethostname(),
            cache_dir,
            int(props.major),
            int(props.minor),
            int(props.multi_processor_count),
            bool(self.runner.is_deepgemm_jit_enabled()),
            int(pipe.token_capacity * pipe.K),
            int(pipe.E),
            int(pipe.H),
            int(self.I),
            bool(pipe.config.fuse_fc1_epilogue),
            self._nvcc_available(compiler),
        )

    def _prepare_gemm_jit_collective(self) -> None:
        pipe = self.pipe
        rank = getattr(pipe, "rank", 0)
        cache_identities = _run_guarded_phase(
            pipe._comm,
            rank,
            "gemm-jit-cache-layout",
            self._gemm_jit_cache_identity,
        )
        local_identity = cache_identities[rank]
        local_group = local_identity[:-1]
        peers = [
            peer_rank
            for peer_rank, identity in enumerate(cache_identities)
            if identity[:-1] == local_group
        ]
        leader_rank = next(
            (peer_rank for peer_rank in peers if cache_identities[peer_rank][-1]),
            peers[0],
        )

        _run_guarded_phase(
            pipe._comm,
            rank,
            "gemm-jit-cache-warm",
            lambda: self._prepare_gemm_jit() if rank == leader_rank else None,
        )
        _run_guarded_phase(
            pipe._comm,
            rank,
            "gemm-jit-cache-load",
            lambda: self._prepare_gemm_jit() if rank != leader_rank else None,
        )

    def _prepare_gemm_jit(self) -> None:
        """Load or compile every DeepGEMM kernel used by this configuration."""
        pipe = self.pipe
        self._require_gemm_jit_runtime()
        offsets0 = torch.zeros(pipe.E + 1, dtype=torch.int64, device=pipe.device)
        if pipe.config.fuse_fc1_epilogue:
            try:
                self.runner.moe_gemm_fc1_fused(
                    self.a2,
                    self.sfa2,
                    self.a1.view(torch.float8_e4m3fn),
                    self.w13_fp8,
                    offsets0,
                    2 * self.I,
                    pipe.H,
                    self.sfa1,
                    self.w13_sf,
                    True,
                )
            except Exception as exc:
                raise RuntimeError(
                    self._FC1_FUSED_FAIL_HELP + f" Underlying error: {exc}"
                ) from exc
        else:
            self.runner.moe_gemm(
                self.h,
                self.a1.view(torch.float8_e4m3fn),
                self.w13_fp8,
                offsets0,
                2 * self.I,
                pipe.H,
                self.sfa1,
                self.w13_sf,
                True,
            )
        self.runner.moe_gemm(
            self.y,
            self.a2.view(torch.float8_e4m3fn),
            self.w2_fp8,
            offsets0,
            pipe.H,
            self.I,
            self.sfa2,
            self.w2_sf,
            True,
        )
        torch.cuda.synchronize()

    def _g_buf(self) -> torch.Tensor:
        if self._g is None:
            # sized off a2 (same m_buf rows); h may be None in fused mode
            self._g = torch.empty(
                self.a2.shape[0], self.I, dtype=torch.bfloat16, device=self.pipe.device
            )
        return self._g

    @property
    def state(self) -> str:
        return self._state.value

    def _require_usable(self) -> None:
        if self._state == _RunnerState.POISONED:
            raise RuntimeError(
                "sm90_push runner is poisoned by an earlier mid-round failure; "
                "rebuild the pipe and runner"
            )
        if self._state == _RunnerState.DESTROYED:
            raise RuntimeError("sm90_push runner has been destroyed")

    def _validate_round_inputs(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> int:
        # CUDA-graph capture requires caller-owned storage; validation rejects
        # layouts that would otherwise need an implicit .to() or .contiguous().
        pipe = self.pipe
        if x.ndim != 2 or x.shape[1] != pipe.H:
            raise ValueError(f"x must be (T, {pipe.H})")
        T = x.shape[0]
        if pipe.token_capacity < T:
            raise ValueError(f"T {T} exceeds token_capacity {pipe.token_capacity}")
        if topk_ids.shape != (T, pipe.K) or topk_weights.shape != (T, pipe.K):
            raise ValueError(f"routing must be (T, {pipe.K})")
        for name, tensor, dtype in (
            ("x", x, torch.bfloat16),
            ("topk_ids", topk_ids, torch.int32),
            ("topk_weights", topk_weights, torch.float32),
        ):
            if tensor.dtype != dtype:
                raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
            if tensor.device != pipe.device:
                raise ValueError(
                    f"{name} must be on {pipe.device}, got {tensor.device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        return T

    def _validate_output(self, output: torch.Tensor, num_tokens: int) -> None:
        pipe = self.pipe
        if output.shape != (num_tokens, pipe.H):
            raise ValueError(
                f"output must be ({num_tokens}, {pipe.H}), got {tuple(output.shape)}"
            )
        if output.dtype != pipe.out_dtype:
            raise ValueError(f"output must be {pipe.out_dtype}, got {output.dtype}")
        if output.device != pipe.device:
            raise ValueError(f"output must be on {pipe.device}, got {output.device}")
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")

    def _current_stream(self) -> tuple[torch.cuda.Stream, bool]:
        stream = torch.cuda.current_stream(self.pipe.device)
        return stream, torch.cuda.is_current_stream_capturing()

    @staticmethod
    def _stream_context(stream: torch.cuda.Stream):
        return torch.cuda.stream(stream)

    def _poison(self) -> None:
        self._state = _RunnerState.POISONED
        self.pipe.proto_abort()
        self._staged_tokens = None
        self._staged_stream = None
        self._staged_stream_capturing = False
        self._staged_output = None

    def stage_inputs(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        output: torch.Tensor,
    ) -> None:
        """Validate one round, then dispatch its BF16 input and routing tensors."""
        self._require_usable()
        if self._state == _RunnerState.STAGED:
            raise RuntimeError(
                "sm90_push stage_inputs called twice for one round; call compute first"
            )
        T = self._validate_round_inputs(x, topk_ids, topk_weights)
        stream, capturing = self._current_stream()
        stream_id = int(stream.cuda_stream)
        if (
            not capturing
            and self._round_event is not None
            and self._round_stream_id != stream_id
            and not self._round_event.query()
        ):
            raise RuntimeError(
                "sm90_push cannot start a round on a different stream while "
                "the previous round is still executing"
            )
        self._validate_output(output, T)

        nv = self.record_stages
        try:
            with _record_stage("begin_round", nv):
                self.pipe.proto_begin_round()
            with _record_stage("dispatch", nv):
                self.pipe.proto_dispatch(x, topk_ids, topk_weights)
        except Exception:
            self._poison()
            raise
        self._staged_tokens = T
        self._staged_stream = stream
        self._staged_stream_capturing = capturing
        self._staged_output = output
        self._state = _RunnerState.STAGED

    def compute(self, *, output: torch.Tensor) -> torch.Tensor:
        """Finish the staged round and reduce directly into the supplied output.

        A cross-stream call hands caller work to the staged stream before launch
        and makes the caller stream wait for the completed round before returning.
        """
        self._require_usable()
        if self._state != _RunnerState.STAGED:
            raise RuntimeError(
                "sm90_push compute requires a preceding stage_inputs call"
            )
        pipe = self.pipe
        T = self._staged_tokens
        staged_stream = self._staged_stream
        staged_capturing = self._staged_stream_capturing
        staged_output = self._staged_output
        assert T is not None
        assert staged_stream is not None
        assert staged_output is not None
        staged_stream_id = int(staged_stream.cuda_stream)
        caller_stream, _caller_capturing = self._current_stream()
        caller_stream_id = int(caller_stream.cuda_stream)
        output_mismatch = output is not staged_output
        streams_differ = caller_stream_id != staged_stream_id
        if streams_differ:
            if self._caller_ready_event is None:
                self._caller_ready_event = torch.cuda.Event()
            self._caller_ready_event.record(caller_stream)
            staged_stream.wait_event(self._caller_ready_event)
        stream_context = (
            contextlib.nullcontext()
            if not streams_differ
            else self._stream_context(staged_stream)
        )
        nv = self.record_stages
        try:
            with stream_context:
                with _record_stage("wait_prefix", nv):
                    pipe.proto_wait_prefix()
                with _record_stage("compact", nv):
                    pipe.proto_compact(self.a1, self.sfa1, self.meta, self.row_expert)
                with _record_stage("fc1", nv):
                    if pipe.config.fuse_fc1_epilogue:
                        try:
                            self.runner.moe_gemm_fc1_fused(
                                self.a2,
                                self.sfa2,
                                self.a1.view(torch.float8_e4m3fn),
                                self.w13_fp8,
                                pipe._offsets,
                                2 * self.I,
                                pipe.H,
                                self.sfa1,
                                self.w13_sf,
                                True,
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                self._FC1_FUSED_FAIL_HELP + f" Underlying error: {exc}"
                            ) from exc
                    else:
                        self.runner.moe_gemm(
                            self.h,
                            self.a1.view(torch.float8_e4m3fn),
                            self.w13_fp8,
                            pipe._offsets,
                            2 * self.I,
                            pipe.H,
                            self.sfa1,
                            self.w13_sf,
                            True,
                        )
                if not pipe.config.fuse_fc1_epilogue:
                    with _record_stage("act_quant", nv):
                        if pipe.config.fuse_act:
                            pipe.proto_silu_mul_quant(
                                self.h, self.a2, self.sfa2, self.row_expert
                            )
                        else:
                            g = self._g_buf()
                            pipe.module.sm90_silu_mul_gated(
                                g, self.h, pipe._m_dev, g.shape[0]
                            )
                            pipe.module.sm90_quant_grouped(
                                self.a2,
                                self.sfa2,
                                g,
                                pipe._offsets,
                                pipe._pad_base,
                                pipe._m_dev,
                                pipe._p_dev,
                                self.row_expert,
                                g.shape[0],
                            )
                with _record_stage("fc2", nv):
                    self.runner.moe_gemm(
                        self.y,
                        self.a2.view(torch.float8_e4m3fn),
                        self.w2_fp8,
                        pipe._offsets,
                        pipe.H,
                        self.I,
                        self.sfa2,
                        self.w2_sf,
                        True,
                    )
                with _record_stage("combine", nv):
                    pipe.proto_combine(self.y, self.meta)
                with _record_stage("wait_combine", nv):
                    pipe.proto_wait_combine()
                with _record_stage("reduce", nv):
                    pipe.proto_reduce(staged_output, T)
                with _record_stage("ack", nv):
                    pipe.proto_ack()
        except Exception:
            self._poison()
            raise

        self._state = _RunnerState.IDLE
        self._staged_tokens = None
        self._staged_stream = None
        self._staged_stream_capturing = False
        self._staged_output = None
        if not staged_capturing or streams_differ:
            if self._round_event is None:
                self._round_event = torch.cuda.Event()
            self._round_event.record(staged_stream)
            self._round_stream_id = staged_stream_id
            if streams_differ:
                caller_stream.wait_event(self._round_event)
        if output_mismatch:
            raise RuntimeError(
                "sm90_push compute must receive the same output tensor that was "
                "validated by stage_inputs; the staged round was completed into the "
                "validated output"
            )
        return staged_output

    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Submit a complete round while preserving the two-phase entry points."""
        self.stage_inputs(x, topk_ids, topk_weights, output=output)
        return self.compute(output=output)

    def abort(self) -> None:
        """Poison the active pipe without entering a cross-rank barrier."""
        if self._state in (_RunnerState.POISONED, _RunnerState.DESTROYED):
            return
        self._poison()

    def destroy(self) -> None:
        """Quiesce local work and make this runner reject new work."""
        if self._state == _RunnerState.DESTROYED:
            return
        if self._state == _RunnerState.STAGED:
            self._poison()
        self.pipe.destroy()
        self._state = _RunnerState.DESTROYED
        self._staged_tokens = None
        self._staged_stream = None
        self._staged_stream_capturing = False
        self._staged_output = None
        self._caller_ready_event = None
        self._round_event = None
        self._workspace = None
        self.runner = None

    __call__ = forward
