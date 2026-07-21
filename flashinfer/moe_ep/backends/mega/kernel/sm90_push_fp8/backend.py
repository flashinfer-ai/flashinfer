"""SM90 push FP8 mega-MoE kernel backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .....config import BootstrapConfig, FleetParams
from .....core.kernel.base import MegaKernelBackend
from .....core.kernel.registry import register_mega_kernel
from .....core.runtime import TORCH_DIST
from .....core.validation.common import (
    MoEEpArchError,
    MoEEpConfigError,
    validate_mega_fleet_params,
)
from .....weights import MoEWeightPack
from .config import Sm90PushFp8MegaMoeConfig
from .staging import validate_sm90_push_fp8_forward_inputs
from .weights import (
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from .....tensors import MoEEpTensors


@dataclass
class _Sm90PushFp8Workspace:
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
            "sm90_push_fp8 requires an SM90 (Hopper) device; "
            f"host has sm_{major}{minor}"
        )


@register_mega_kernel("sm90_push_fp8")
class Sm90PushFp8MegaKernelBackend(MegaKernelBackend):
    def __init__(self, config: Sm90PushFp8MegaMoeConfig) -> None:
        if not isinstance(config, Sm90PushFp8MegaMoeConfig):
            raise TypeError(
                "sm90_push_fp8 config must be Sm90PushFp8MegaMoeConfig, got "
                f"{type(config).__name__}"
            )
        super().__init__(config)
        self._kernel_config = config

    @classmethod
    def kernel_name(cls) -> str:
        return "sm90_push_fp8"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        del bootstrap
        return frozenset({TORCH_DIST})

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        _validate_sm90_arch()
        kcfg = self._kernel_config
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=kcfg.intermediate_size,
            top_k=kcfg.top_k,
        )
        if bootstrap.world_size > 32:
            raise MoEEpConfigError(
                "sm90_push_fp8 supports a single-node EP group of at most 32 "
                f"ranks, got world_size={bootstrap.world_size}"
            )
        if kcfg.top_k not in (1, 2, 4, 6, 8):
            raise MoEEpConfigError(
                f"sm90_push_fp8 top_k must be one of (1, 2, 4, 6, 8), got {kcfg.top_k}"
            )
        if not kcfg.fuse_fc1_epilogue and kcfg.intermediate_size > 16384:
            raise MoEEpConfigError(
                "sm90_push_fp8 unfused silu_mul_quant uses shared-memory staging "
                "and requires intermediate_size <= 16384, got "
                f"{kcfg.intermediate_size}; set fuse_fc1_epilogue=True to use "
                "the fused FC1 path for larger intermediate sizes"
            )
        if kcfg.payload_dtype not in ("fp8", "bf16"):
            raise MoEEpConfigError(
                "sm90_push_fp8 payload_dtype must be 'fp8' or 'bf16', got "
                f"{kcfg.payload_dtype!r}"
            )
        if kcfg.combine_dtype not in ("fp8", "bf16"):
            raise MoEEpConfigError(
                "sm90_push_fp8 combine_dtype must be 'fp8' or 'bf16', got "
                f"{kcfg.combine_dtype!r}"
            )
        if kcfg.grouped_combine and kcfg.combine_dtype != "fp8":
            raise MoEEpConfigError(
                "sm90_push_fp8 grouped_combine requires combine_dtype='fp8'"
            )
        try:
            capacity_factor = float(kcfg.capacity_factor)
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(
                "sm90_push_fp8 capacity_factor must be finite and in (0, 1]"
            ) from exc
        if not math.isfinite(capacity_factor) or not (0.0 < capacity_factor <= 1.0):
            raise MoEEpConfigError(
                "sm90_push_fp8 capacity_factor must be finite and in (0, 1], got "
                f"{kcfg.capacity_factor!r}"
            )
        try:
            timeout_s = float(kcfg.init_timeout_s)
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(
                "sm90_push_fp8 init_timeout_s must be finite and positive"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise MoEEpConfigError(
                "sm90_push_fp8 init_timeout_s must be finite and positive, got "
                f"{kcfg.init_timeout_s!r}"
            )
        if bootstrap.stream != 0:
            raise MoEEpConfigError(
                "sm90_push_fp8 launches on the current torch CUDA stream; "
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

    def _allocate_workspace(self, fleet_params: FleetParams) -> _Sm90PushFp8Workspace:
        from .....kernel_src.sm90_push_megamoe import (
            Sm90PushCombine,
            Sm90PushConfig,
            Sm90PushMoERunner,
            Sm90PushPayload,
            Sm90PushPipe,
        )
        from ......comm.mnnvl import TorchDistBackend

        transformed_weights = self._transformed_weights
        if transformed_weights is None:
            raise RuntimeError(
                "sm90_push_fp8 weights must be preprocessed or validated before "
                "workspace allocation"
            )
        kcfg = self._kernel_config
        comm = TorchDistBackend(group=self.ep_comm_group)
        timeout_s = float(kcfg.init_timeout_s)
        timeout_error = None
        try:
            comm.set_timeout(timeout_s)
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
                "sm90_push_fp8 failed to configure the EP process-group timeout: "
                + " | ".join(timeout_failures)
            )
        if any(peer_timeout != timeout_s for peer_timeout, _error in timeout_reports):
            raise RuntimeError(
                "sm90_push_fp8 init_timeout_s must match on every EP rank; got "
                f"{[peer_timeout for peer_timeout, _error in timeout_reports]}"
            )
        pipe = Sm90PushPipe(
            ep_size=self.ep_world_size,
            rank=self.ep_rank,
            num_local_experts=fleet_params.num_experts // self.ep_world_size,
            hidden_size=fleet_params.token_hidden_size,
            top_k=kcfg.top_k,
            token_capacity=fleet_params.max_tokens_per_rank,
            device_index=torch.cuda.current_device(),
            config=Sm90PushConfig(
                payload_dtype=Sm90PushPayload(kcfg.payload_dtype),
                combine_dtype=Sm90PushCombine(kcfg.combine_dtype),
                fuse_act=True,
                capacity_factor=float(kcfg.capacity_factor),
                dedup_dispatch=kcfg.dedup_dispatch,
                grouped_combine=kcfg.grouped_combine,
                fuse_fc1_epilogue=kcfg.fuse_fc1_epilogue,
            ),
            comm_backend=comm,
            out_dtype=torch.bfloat16,
            allow_unverified_p2p=kcfg.allow_unverified_p2p,
        )
        try:
            runner = Sm90PushMoERunner(pipe, transformed_weights)
        except Exception:
            pipe.destroy()
            raise
        return _Sm90PushFp8Workspace(
            pipe=pipe,
            runner=runner,
            transformed_weights=transformed_weights,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_sm90_push_fp8_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    @staticmethod
    def _live_workspace(workspace: Any) -> _Sm90PushFp8Workspace:
        if not isinstance(workspace, _Sm90PushFp8Workspace):
            raise TypeError(
                "sm90_push_fp8 workspace must be created by this backend, got "
                f"{type(workspace).__name__}"
            )
        if workspace.destroyed:
            raise RuntimeError("sm90_push_fp8 workspace has been destroyed")
        if workspace.poisoned:
            raise RuntimeError(
                "sm90_push_fp8 workspace is poisoned by an earlier round failure"
            )
        return workspace

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
        output: torch.Tensor | None = None,
    ) -> None:
        if not quantize_input:
            raise MoEEpConfigError(
                "sm90_push_fp8 requires MegaConfig.quantize_input=True"
            )
        ws = self._live_workspace(workspace)
        if ws.transformed_weights is not self._transformed_weights:
            raise RuntimeError(
                "sm90_push_fp8 backend weights differ from the workspace bundle; "
                "rebuild the layer after replacing transformed weights"
            )
        if output is None:
            raise MoEEpConfigError(
                "sm90_push_fp8 stage_inputs requires the destination output tensor"
            )
        try:
            ws.runner.stage_inputs(
                t.hidden_states,
                t.topk_ids,
                t.topk_weights,
                output=output,
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
        num_tokens = ws.staged_tokens
        if num_tokens is None:
            raise RuntimeError(
                "sm90_push_fp8 compute() requires a successful stage_inputs()"
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
                "sm90_push_fp8 runner must return the caller-provided output tensor"
            )
        if weights_mismatch:
            raise RuntimeError(
                "sm90_push_fp8 compute received a different weight bundle; the "
                "staged round completed with the weights bound to its workspace"
            )
        return output

    def destroy(self, workspace: Any) -> None:
        if workspace is None:
            return
        if not isinstance(workspace, _Sm90PushFp8Workspace):
            raise TypeError(
                "sm90_push_fp8 workspace must be created by this backend, got "
                f"{type(workspace).__name__}"
            )
        if workspace.destroyed:
            return
        workspace.runner.destroy()
        workspace.destroyed = True
        workspace.staged_tokens = None


__all__ = ["Sm90PushFp8MegaKernelBackend"]
