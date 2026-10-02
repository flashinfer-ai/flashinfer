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

SM90 push BF16 mega-MoE kernel backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import TORCH_DIST
from ......core.validation.common import (
    MoEEpArchError,
    MoEEpConfigError,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .config import Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig
from .staging import validate_sm90_push_bf16_forward_inputs
from .weights import preprocess_mega_weights, validate_transformed_mega_weights

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


@dataclass
class _Sm90PushBf16Workspace:
    pipe: Any
    runner: Any
    transformed_weights: Any
    staged_tokens: int | None = None
    poisoned: bool = False
    destroyed: bool = False


def _validate_sm90_arch() -> None:
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major != 9:
        raise MoEEpArchError(
            "sm90_push_bf16 requires an SM90 (Hopper) device; "
            f"host has sm_{major}{minor}"
        )


def _set_process_group_timeout(group: Any, timeout_s: float) -> None:
    import torch.distributed as dist

    timeout = timedelta(seconds=timeout_s)
    set_pg_timeout = getattr(dist, "set_timeout", None)
    if set_pg_timeout is None:
        distributed_c10d = getattr(dist, "distributed_c10d", None)
        set_pg_timeout = getattr(distributed_c10d, "_set_pg_timeout", None)
    if set_pg_timeout is None:
        raise RuntimeError(
            "torch.distributed exposes neither set_timeout nor "
            "distributed_c10d._set_pg_timeout"
        )
    set_pg_timeout(timeout, group)


@register_mega_kernel(
    "sm90_bf16_bf16_bf16_push_cuda",
    deprecated_aliases=("sm90_push_bf16",),
)
class Sm90PushBf16MegaKernelBackend(MegaKernelBackend):
    def __init__(self, config: Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig) -> None:
        if not isinstance(config, Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig):
            raise TypeError(
                "sm90_bf16_bf16_bf16_push_cuda config must be "
                "Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig, got "
                f"{type(config).__name__}"
            )
        super().__init__(config)
        self._kernel_config = config

    @classmethod
    def kernel_name(cls) -> str:
        return "sm90_bf16_bf16_bf16_push_cuda"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        del bootstrap
        return frozenset({TORCH_DIST})

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        _validate_sm90_arch()
        config = self._kernel_config
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=config.intermediate_size,
            top_k=config.top_k,
        )
        if bootstrap.world_size > 32:
            raise MoEEpConfigError(
                "sm90_push_bf16 supports a single-node EP group of at most 32 "
                f"ranks, got world_size={bootstrap.world_size}"
            )
        if config.top_k not in (1, 2, 4, 6, 8):
            raise MoEEpConfigError(
                "sm90_push_bf16 top_k must be one of (1, 2, 4, 6, 8), got "
                f"{config.top_k}"
            )
        if config.intermediate_size <= 0 or config.intermediate_size % 128:
            raise MoEEpConfigError(
                "sm90_push_bf16 intermediate_size must be a positive multiple "
                f"of 128, got {config.intermediate_size}"
            )
        if config.wave_schedule not in ("mono", "serial2", "pipe2"):
            raise MoEEpConfigError(
                "sm90_push_bf16 wave_schedule must be one of "
                "('mono', 'serial2', 'pipe2'), got "
                f"{config.wave_schedule!r}"
            )
        try:
            capacity_factor = float(config.capacity_factor)
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(
                "sm90_push_bf16 capacity_factor must be finite and in (0, 1]"
            ) from exc
        if not math.isfinite(capacity_factor) or not (0.0 < capacity_factor <= 1.0):
            raise MoEEpConfigError(
                "sm90_push_bf16 capacity_factor must be finite and in (0, 1], got "
                f"{config.capacity_factor!r}"
            )
        if config.wave_schedule != "mono" and capacity_factor != 1.0:
            raise MoEEpConfigError(
                "sm90_push_bf16 two-wave schedules require capacity_factor=1.0 "
                "because each wave owns an independent receive window"
            )
        try:
            timeout_s = float(config.init_timeout_s)
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(
                "sm90_push_bf16 init_timeout_s must be finite and positive"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise MoEEpConfigError(
                "sm90_push_bf16 init_timeout_s must be finite and positive, got "
                f"{config.init_timeout_s!r}"
            )
        if bootstrap.stream != 0:
            raise MoEEpConfigError(
                "sm90_push_bf16 launches on the current torch CUDA stream; "
                "BootstrapConfig.stream must be 0"
            )

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> Any:
        transformed = preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            num_local_experts=fleet_params.num_experts // self.ep_world_size,
            fuse_fc1_epilogue=self._kernel_config.fuse_fc1_epilogue,
        )
        self._transformed_weights = transformed
        return transformed

    def validate_transformed_weights(
        self,
        transformed_weights: Any,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        del bootstrap
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            num_local_experts=fleet_params.num_experts // self.ep_world_size,
            fuse_fc1_epilogue=self._kernel_config.fuse_fc1_epilogue,
        )
        self._transformed_weights = transformed_weights

    def _allocate_workspace(self, fleet_params: FleetParams) -> _Sm90PushBf16Workspace:
        from ......kernel_src.sm90.push_style_megamoe import (
            Sm90PushBf16MoERunner,
            Sm90PushBf16TwoWaveRunner,
            Sm90PushCombine,
            Sm90PushConfig,
            Sm90PushPayload,
            Sm90PushPipe,
        )
        from .......comm.mnnvl import TorchDistBackend

        transformed_weights = self._transformed_weights
        if transformed_weights is None:
            raise RuntimeError(
                "sm90_push_bf16 weights must be prepared before workspace allocation"
            )
        config = self._kernel_config
        comm = TorchDistBackend(group=self.ep_comm_group)
        timeout_s = float(config.init_timeout_s)
        timeout_error = None
        try:
            _set_process_group_timeout(self.ep_comm_group, timeout_s)
        except Exception as exc:  # noqa: BLE001 - report the failure on every EP rank
            timeout_error = f"{type(exc).__name__}: {exc}"
        timeout_reports = comm.allgather((timeout_s, timeout_error))
        timeout_failures = [
            f"rank {rank}: {error}"
            for rank, (_timeout, error) in enumerate(timeout_reports)
            if error is not None
        ]
        if timeout_failures:
            raise RuntimeError(
                "sm90_push_bf16 failed to configure the EP timeout: "
                + " | ".join(timeout_failures)
            )
        if any(peer_timeout != timeout_s for peer_timeout, _ in timeout_reports):
            raise RuntimeError(
                "sm90_push_bf16 init_timeout_s must match on every EP rank"
            )
        pipe_config = Sm90PushConfig(
            payload_dtype=Sm90PushPayload.BF16,
            combine_dtype=Sm90PushCombine.BF16,
            fuse_act=False,
            capacity_factor=float(config.capacity_factor),
            dedup_dispatch=config.dedup_dispatch,
            grouped_combine=config.grouped_combine,
            fuse_fc1_epilogue=config.fuse_fc1_epilogue,
        )

        def _make_pipe(token_capacity: int) -> Sm90PushPipe:
            return Sm90PushPipe(
                ep_size=self.ep_world_size,
                rank=self.ep_rank,
                num_local_experts=fleet_params.num_experts // self.ep_world_size,
                hidden_size=fleet_params.token_hidden_size,
                top_k=config.top_k,
                token_capacity=token_capacity,
                device_index=torch.cuda.current_device(),
                config=pipe_config,
                comm_backend=comm,
                out_dtype=torch.bfloat16,
                allow_unverified_p2p=config.allow_unverified_p2p,
            )

        pipe = None
        pipe1 = None
        runner: Sm90PushBf16MoERunner | Sm90PushBf16TwoWaveRunner
        try:
            if config.wave_schedule == "mono":
                pipe = _make_pipe(fleet_params.max_tokens_per_rank)
                runner = Sm90PushBf16MoERunner(pipe, transformed_weights)
            else:
                wave_capacity = (fleet_params.max_tokens_per_rank + 1) // 2
                pipe = _make_pipe(wave_capacity)
                pipe1 = _make_pipe(wave_capacity)
                runner = Sm90PushBf16TwoWaveRunner(
                    pipe,
                    pipe1,
                    transformed_weights,
                    schedule=config.wave_schedule,
                )
        except Exception:
            if pipe1 is not None:
                pipe1.destroy()
            if pipe is not None:
                pipe.destroy()
            raise
        assert pipe is not None
        return _Sm90PushBf16Workspace(pipe, runner, transformed_weights)

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_sm90_push_bf16_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    @staticmethod
    def _live_workspace(workspace: Any) -> _Sm90PushBf16Workspace:
        if not isinstance(workspace, _Sm90PushBf16Workspace):
            raise TypeError("sm90_push_bf16 workspace must be created by this backend")
        if workspace.destroyed:
            raise RuntimeError("sm90_push_bf16 workspace has been destroyed")
        if workspace.poisoned:
            raise RuntimeError(
                "sm90_push_bf16 workspace is poisoned by an earlier failure"
            )
        return workspace

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        if not quantize_input:
            raise MoEEpConfigError(
                "sm90_push_bf16 requires MegaConfig.quantize_input=True"
            )
        ws = self._live_workspace(workspace)
        if ws.transformed_weights is not self._transformed_weights:
            raise RuntimeError(
                "sm90_push_bf16 backend weights differ from the workspace bundle"
            )
        try:
            ws.runner.stage_inputs(
                t.hidden_states,
                t.topk_ids,
                t.topk_weights,
            )
        except Exception:
            if ws.runner.state == "poisoned":
                ws.poisoned = True
            raise
        ws.staged_tokens = t.num_tokens

    def compute(
        self,
        workspace: Any,
        transformed_weights: Any,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        ws = self._live_workspace(workspace)
        weights_mismatch = (
            transformed_weights is not ws.transformed_weights
            or transformed_weights is not self._transformed_weights
        )
        if ws.staged_tokens is None:
            raise RuntimeError(
                "sm90_push_bf16 compute requires a successful stage_inputs"
            )
        try:
            result = ws.runner.compute(output=output)
        except Exception:
            if ws.runner.state == "poisoned":
                ws.poisoned = True
            raise
        finally:
            ws.staged_tokens = None
        if result is not output:
            raise RuntimeError(
                "sm90_push_bf16 runner must return the caller-provided output"
            )
        if weights_mismatch:
            raise RuntimeError(
                "sm90_push_bf16 compute received a different weight bundle; the "
                "staged round completed with the weights bound to its workspace"
            )
        return output

    def destroy(self, workspace: Any) -> None:
        if workspace is None:
            return
        if not isinstance(workspace, _Sm90PushBf16Workspace):
            raise TypeError("sm90_push_bf16 workspace must be created by this backend")
        if workspace.destroyed:
            return
        workspace.runner.destroy()
        workspace.destroyed = True
        workspace.staged_tokens = None


__all__ = ["Sm90PushBf16MegaKernelBackend"]
