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

SM90 push runner for BF16 expert weights and activations.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .bf16_gemm import create_sm90_push_bf16_gemm_runner
from .bf16_tactics import Bf16GemmTactic, estimate_sm90_push_bf16_expected_m
from .bf16_weights import Sm90PushBf16Weights
from .protocol import (
    Sm90PushCombine,
    Sm90PushPayload,
    Sm90PushPipe,
    _run_guarded_phase,
)
from .runner import Sm90PushMoERunner


class _Bf16Fc1Runner(Protocol):
    def run(
        self,
        output: torch.Tensor,
        activation: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor: ...

    def tactic_provenance(self) -> dict[str, object]: ...


class _Bf16GemmRunner(_Bf16Fc1Runner, Protocol):
    def run(
        self,
        output: torch.Tensor,
        activation: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        *,
        prepare_schedule: bool = True,
    ) -> torch.Tensor: ...


def _align(value: int, alignment: int = 128) -> int:
    return (value + alignment - 1) // alignment * alignment


def _normalize_gemm_implementation(implementation: str) -> str:
    implementation = str(implementation).lower()
    if implementation not in ("cutlass_prepared", "persistent_offsets"):
        raise ValueError(
            "gemm_implementation must be 'cutlass_prepared' or "
            f"'persistent_offsets', got {implementation!r}"
        )
    return implementation


class Sm90PushBf16MoERunner(Sm90PushMoERunner):
    """Two-phase SM90 push runner for canonical BF16 expert weights."""

    fc1: _Bf16Fc1Runner | None
    fc2: _Bf16GemmRunner | None

    def __init__(
        self,
        pipe: Sm90PushPipe,
        weights: Sm90PushBf16Weights,
        *,
        fc1_gemm_tactic: Bf16GemmTactic | str = "production",
        fc2_gemm_tactic: Bf16GemmTactic | str = "production",
        gemm_implementation: str = "cutlass_prepared",
    ) -> None:
        if not isinstance(weights, Sm90PushBf16Weights):
            raise TypeError("weights must be Sm90PushBf16Weights")
        self._init_round_state(pipe)
        self.weights = weights

        def _local_init() -> None:
            if pipe.config.payload_dtype is not Sm90PushPayload.BF16:
                raise ValueError("SM90 push BF16 requires BF16 dispatch wire")
            if pipe.config.combine_dtype is not Sm90PushCombine.BF16:
                raise ValueError("SM90 push BF16 requires BF16 combine wire")
            if pipe.config.fuse_act:
                raise ValueError("SM90 push BF16 uses the BF16 activation path")
            self._init_bf16(
                weights,
                fc1_gemm_tactic,
                fc2_gemm_tactic,
                _normalize_gemm_implementation(gemm_implementation),
            )

        _run_guarded_phase(
            pipe._comm,
            getattr(pipe, "rank", 0),
            "bf16-weights+gemm-resources",
            _local_init,
        )

    def _init_bf16(
        self,
        weights: Sm90PushBf16Weights,
        fc1_gemm_tactic: Bf16GemmTactic | str,
        fc2_gemm_tactic: Bf16GemmTactic | str,
        gemm_implementation: str,
    ) -> None:
        pipe = self.pipe
        self.gemm_implementation = gemm_implementation
        experts, two_intermediate, hidden = weights.w13.shape
        if experts != pipe.E or hidden != pipe.H:
            raise ValueError(
                f"SM90 push BF16 w13 must have shape (E={pipe.E}, 2I, H={pipe.H})"
            )
        self.I = two_intermediate // 2
        if tuple(weights.w2.shape) != (pipe.E, pipe.H, self.I):
            raise ValueError(
                f"SM90 push BF16 w2 must have shape ({pipe.E}, {pipe.H}, {self.I})"
            )
        if weights.w13.device != pipe.device:
            raise ValueError("SM90 push BF16 weights must be on the pipe device")

        self._padded_max_rows = _align(pipe.m_cap + 7 * pipe.E)
        actual_rows = _align(pipe.m_cap)
        device = pipe.device
        self.a1 = torch.empty(
            self._padded_max_rows,
            pipe.H,
            dtype=torch.bfloat16,
            device=device,
        )
        self.meta = torch.empty(actual_rows, 4, dtype=torch.int32, device=device)
        self.a2 = torch.empty(
            self._padded_max_rows,
            self.I,
            dtype=torch.bfloat16,
            device=device,
        )
        self.y = torch.empty(
            self._padded_max_rows,
            pipe.H,
            dtype=torch.bfloat16,
            device=device,
        )
        self.real_to_padded = torch.empty(
            actual_rows,
            dtype=torch.int32,
            device=device,
        )
        self.bf16_offsets = torch.empty(
            pipe.E + 1,
            dtype=torch.int64,
            device=device,
        )
        self.bf16_tile_prefix = torch.empty_like(self.bf16_offsets)
        self.bf16_m_dev = torch.zeros(1, dtype=torch.int32, device=device)
        expected_m = estimate_sm90_push_bf16_expected_m(
            token_capacity=pipe.token_capacity,
            top_k=pipe.K,
            num_local_experts=pipe.E,
        )
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        if pipe.config.fuse_fc1_epilogue:
            from .bf16_fc1_fused import create_sm90_push_bf16_fused_fc1_runner

            if not (
                isinstance(fc1_gemm_tactic, str)
                and fc1_gemm_tactic.lower() in ("auto", "production")
            ):
                raise ValueError(
                    "fc1_gemm_tactic does not apply to the fused FC1 kernel"
                )
            self.h = None
            self.fc1 = create_sm90_push_bf16_fused_fc1_runner(
                max_rows=self._padded_max_rows,
                num_experts=pipe.E,
                intermediate_size=self.I,
                k=pipe.H,
                device=device,
            )
            shared_schedule_workspace = None
        else:
            self.h = torch.empty(
                self._padded_max_rows,
                two_intermediate,
                dtype=torch.bfloat16,
                device=device,
            )
            if gemm_implementation == "cutlass_prepared":
                fc1 = create_sm90_push_bf16_gemm_runner(
                    max_rows=self._padded_max_rows,
                    num_experts=pipe.E,
                    n=two_intermediate,
                    k=pipe.H,
                    device=device,
                    tactic=fc1_gemm_tactic,
                    expected_m=expected_m,
                    sm_count=sm_count,
                )
                self.fc1 = fc1
                shared_schedule_workspace = fc1.schedule_workspace
            else:
                from .bf16_persistent_gemm import (
                    create_sm90_push_bf16_persistent_gemm_runner,
                )

                self.fc1 = create_sm90_push_bf16_persistent_gemm_runner(
                    max_rows=self._padded_max_rows,
                    num_experts=pipe.E,
                    n=two_intermediate,
                    k=pipe.H,
                    device=device,
                    tactic=fc1_gemm_tactic,
                    expected_m=expected_m,
                    sm_count=sm_count,
                    trusted_offsets=True,
                )
                shared_schedule_workspace = None
        if gemm_implementation == "cutlass_prepared":
            self.fc2 = create_sm90_push_bf16_gemm_runner(
                max_rows=self._padded_max_rows,
                num_experts=pipe.E,
                n=pipe.H,
                k=self.I,
                device=device,
                shared_schedule_workspace=shared_schedule_workspace,
                tactic=fc2_gemm_tactic,
                expected_m=expected_m,
                sm_count=sm_count,
            )
        else:
            from .bf16_persistent_gemm import (
                create_sm90_push_bf16_persistent_gemm_runner,
            )

            self.fc2 = create_sm90_push_bf16_persistent_gemm_runner(
                max_rows=self._padded_max_rows,
                num_experts=pipe.E,
                n=pipe.H,
                k=self.I,
                device=device,
                tactic=fc2_gemm_tactic,
                expected_m=expected_m,
                sm_count=sm_count,
                trusted_offsets=True,
            )

    def _round_compact(self) -> None:
        self.pipe.proto_compact_bf16_padded(
            self.a1,
            self.meta,
            self.real_to_padded,
            self.bf16_offsets,
            self.bf16_tile_prefix,
            self.bf16_m_dev,
            128,
        )

    def _round_fc1(self) -> None:
        fc1 = self.fc1
        if fc1 is None:
            raise RuntimeError("SM90 push BF16 FC1 resources have been released")
        if self.pipe.config.fuse_fc1_epilogue:
            fc1.run(
                self.a2,
                self.a1,
                self.weights.w13,
                self.bf16_offsets,
            )
        else:
            assert self.h is not None
            fc1.run(
                self.h,
                self.a1,
                self.weights.w13,
                self.bf16_offsets,
            )

    def _round_activation(self) -> None:
        assert self.h is not None
        self.pipe.module.sm90_silu_mul_gated(
            self.a2,
            self.h,
            self.bf16_m_dev,
            self.a2.shape[0],
        )

    def _round_activation_stage(self) -> str | None:
        return None if self.pipe.config.fuse_fc1_epilogue else "activation"

    def _round_fc2(self) -> None:
        fc2 = self.fc2
        if fc2 is None:
            raise RuntimeError("SM90 push BF16 FC2 resources have been released")
        fc2.run(
            self.y,
            self.a2,
            self.weights.w2,
            self.bf16_offsets,
            prepare_schedule=self.pipe.config.fuse_fc1_epilogue,
        )

    def _round_combine(self) -> None:
        if self.pipe.config.grouped_combine:
            self.pipe.proto_combine_bf16_grouped_mapped(
                self.y,
                self.meta,
                self.real_to_padded,
            )
        else:
            self.pipe.proto_combine_mapped(self.y, self.meta, self.real_to_padded)

    def tactic_provenance(self) -> dict[str, object]:
        fc1 = self.fc1
        fc2 = self.fc2
        if fc1 is None or fc2 is None:
            raise RuntimeError("SM90 push BF16 GEMM resources have been released")
        return {
            "fused_fc1_epilogue": self.pipe.config.fuse_fc1_epilogue,
            "gemm_implementation": self.gemm_implementation,
            "fc1": fc1.tactic_provenance(),
            "fc2": fc2.tactic_provenance(),
        }

    def _release_resources(self) -> None:
        self.fc1 = None
        self.fc2 = None


__all__ = ["Sm90PushBf16MoERunner"]
