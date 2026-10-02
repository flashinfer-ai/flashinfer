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

Two-wave scheduling for the SM90 push BF16 runner.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum

import torch

from .bf16_runner import Sm90PushBf16MoERunner
from .bf16_weights import Sm90PushBf16Weights
from .protocol import Sm90PushPipe, _record_stage


class _WaveState(Enum):
    IDLE = "idle"
    STAGED = "staged"
    POISONED = "poisoned"
    DESTROYED = "destroyed"


@dataclass(frozen=True)
class _Wave:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


class Sm90PushBf16TwoWaveRunner:
    """Run two isolated push rounds with optional ingress/compute overlap."""

    def __init__(
        self,
        pipe0: Sm90PushPipe,
        pipe1: Sm90PushPipe,
        weights: Sm90PushBf16Weights,
        *,
        schedule: str,
    ) -> None:
        if schedule not in ("serial2", "pipe2"):
            raise ValueError("two-wave schedule must be 'serial2' or 'pipe2'")
        if pipe0.device != pipe1.device:
            raise ValueError("two-wave pipes must use the same CUDA device")
        if (
            pipe0.H,
            pipe0.K,
            pipe0.E,
            pipe0.token_capacity,
            pipe0.out_dtype,
        ) != (
            pipe1.H,
            pipe1.K,
            pipe1.E,
            pipe1.token_capacity,
            pipe1.out_dtype,
        ):
            raise ValueError("two-wave pipes must have identical geometry")

        self.pipe = pipe0
        self._pipe1 = pipe1
        self._schedule = schedule
        self._runner0 = Sm90PushBf16MoERunner(pipe0, weights)
        try:
            self._runner1 = Sm90PushBf16MoERunner(pipe1, weights)
        except Exception:
            self._runner0.destroy()
            raise

        self._state = _WaveState.IDLE
        self._wave0: _Wave | None = None
        self._wave1: _Wave | None = None
        self._caller_stream_id: int | None = None
        self._record_stages = False

        device = pipe0.device
        self._stream0 = torch.cuda.Stream(device=device)
        self._stream1 = torch.cuda.Stream(device=device)
        self._ready = torch.cuda.Event()
        self._dispatch0_done = torch.cuda.Event()
        self._wave0_done = torch.cuda.Event()
        self._final_done = torch.cuda.Event()

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def record_stages(self) -> bool:
        return self._record_stages

    @record_stages.setter
    def record_stages(self, enabled: bool) -> None:
        self._record_stages = bool(enabled)
        if hasattr(self, "_runner0"):
            self._runner0.record_stages = self._record_stages
            self._runner1.record_stages = self._record_stages

    def _require_idle(self) -> None:
        if self._state == _WaveState.STAGED:
            raise RuntimeError("two-wave runner already has a staged round")
        if self._state == _WaveState.POISONED:
            raise RuntimeError("two-wave runner is poisoned by an earlier failure")
        if self._state == _WaveState.DESTROYED:
            raise RuntimeError("two-wave runner has been destroyed")

    def _validate_output(self, output: torch.Tensor, tokens: int) -> None:
        if output.shape != (tokens, self.pipe.H):
            raise ValueError(f"output must be ({tokens}, {self.pipe.H})")
        if output.dtype != self.pipe.out_dtype:
            raise ValueError(
                f"output must be {self.pipe.out_dtype}, got {output.dtype}"
            )
        if output.device != self.pipe.device:
            raise ValueError(
                f"output must be on {self.pipe.device}, got {output.device}"
            )
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")

    def _poison(self) -> None:
        self._state = _WaveState.POISONED
        for runner in (self._runner0, self._runner1):
            with contextlib.suppress(Exception):
                runner.abort()
        self._clear_staged()

    def _clear_staged(self) -> None:
        self._wave0 = None
        self._wave1 = None

    def stage_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Stage wave zero and retain immutable views for wave one."""
        self._require_idle()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "two-wave scheduling does not support CUDA graph capture"
            )

        tokens = hidden_states.shape[0]
        if tokens > 2 * self.pipe.token_capacity:
            raise ValueError(
                f"T {tokens} exceeds two-wave capacity {2 * self.pipe.token_capacity}"
            )
        if topk_ids.shape != (tokens, self.pipe.K):
            raise ValueError(f"topk_ids must be ({tokens}, {self.pipe.K})")
        if topk_weights.shape != (tokens, self.pipe.K):
            raise ValueError(f"topk_weights must be ({tokens}, {self.pipe.K})")
        split = (tokens + 1) // 2
        wave0 = _Wave(
            hidden_states.narrow(0, 0, split),
            topk_ids.narrow(0, 0, split),
            topk_weights.narrow(0, 0, split),
        )
        wave1 = _Wave(
            hidden_states.narrow(0, split, tokens - split),
            topk_ids.narrow(0, split, tokens - split),
            topk_weights.narrow(0, split, tokens - split),
        )

        caller = torch.cuda.current_stream(self.pipe.device)
        caller_id = int(caller.cuda_stream)
        if (
            self._caller_stream_id is not None
            and self._caller_stream_id != caller_id
            and not self._final_done.query()
        ):
            raise RuntimeError(
                "two-wave runner cannot switch caller streams while a round is active"
            )

        self._ready.record(caller)
        self._stream0.wait_event(self._ready)
        self._stream1.wait_event(self._ready)
        try:
            with torch.cuda.stream(self._stream0):
                with _record_stage("wave0_stage", self.record_stages):
                    self._runner0.stage_inputs(
                        wave0.hidden_states,
                        wave0.topk_ids,
                        wave0.topk_weights,
                    )
                self._dispatch0_done.record(self._stream0)
        except Exception:
            self._poison()
            raise

        self._wave0 = wave0
        self._wave1 = wave1
        self._caller_stream_id = caller_id
        self._state = _WaveState.STAGED

    def compute(self, *, output: torch.Tensor) -> torch.Tensor:
        """Finish both waves and join their completion onto the caller stream."""
        if self._state != _WaveState.STAGED:
            self._require_idle()
            raise RuntimeError("two-wave compute requires a preceding stage_inputs")
        if torch.cuda.is_current_stream_capturing():
            self._poison()
            raise RuntimeError(
                "two-wave scheduling does not support CUDA graph capture"
            )
        wave0 = self._wave0
        wave1 = self._wave1
        assert wave0 is not None
        assert wave1 is not None
        tokens0 = wave0.hidden_states.shape[0]
        tokens1 = wave1.hidden_states.shape[0]
        try:
            self._validate_output(output, tokens0 + tokens1)
        except Exception:
            self._poison()
            raise
        output0 = output.narrow(0, 0, tokens0)
        output1 = output.narrow(0, tokens0, tokens1)

        caller = torch.cuda.current_stream(self.pipe.device)
        if int(caller.cuda_stream) != self._caller_stream_id:
            self._poison()
            raise RuntimeError(
                "two-wave stage_inputs and compute must use one caller stream"
            )

        try:
            with torch.cuda.stream(self._stream0):
                with _record_stage("wave0_compute", self.record_stages):
                    self._runner0.compute(output=output0)
                self._wave0_done.record(self._stream0)

            with torch.cuda.stream(self._stream1):
                if self._schedule == "serial2":
                    self._stream1.wait_event(self._wave0_done)
                else:
                    self._stream1.wait_event(self._dispatch0_done)
                with _record_stage("wave1_stage", self.record_stages):
                    self._runner1.stage_inputs(
                        wave1.hidden_states,
                        wave1.topk_ids,
                        wave1.topk_weights,
                    )
                if self._schedule == "pipe2":
                    self._stream1.wait_event(self._wave0_done)
                with _record_stage("wave1_compute", self.record_stages):
                    self._runner1.compute(output=output1)
                self._final_done.record(self._stream1)
            caller.wait_event(self._final_done)
        except Exception:
            self._poison()
            raise

        self._state = _WaveState.IDLE
        self._clear_staged()
        return output

    def abort(self) -> None:
        if self._state in (_WaveState.POISONED, _WaveState.DESTROYED):
            return
        self._poison()

    def tactic_provenance(self) -> dict[str, object]:
        return self._runner0.tactic_provenance()

    def destroy(self) -> None:
        if self._state == _WaveState.DESTROYED:
            return
        if self._state == _WaveState.STAGED:
            self._poison()
        for runner in (self._runner1, self._runner0):
            runner.destroy()
        self._state = _WaveState.DESTROYED
        self._clear_staged()


__all__ = ["Sm90PushBf16TwoWaveRunner"]
