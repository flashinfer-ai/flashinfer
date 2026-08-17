"""SM90 push NVFP4 mega-MoE kernel backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
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
from .config import Sm90PushNvFp4MegaMoeConfig
from .staging import validate_sm90_push_nvfp4_forward_inputs
from .weights import preprocess_mega_weights, validate_transformed_mega_weights

if TYPE_CHECKING:
    from .....tensors import MoEEpTensors


@dataclass
class _Sm90PushNvFp4Workspace:
    pipe: Any
    runner: Any
    active_weights: Any | None = None
    staged_weights: Any | None = None
    staged_tokens: int | None = None
    poisoned: bool = False
    destroyed: bool = False

    def destroy(self) -> None:
        if self.destroyed:
            return
        self.runner.destroy()
        self.active_weights = None
        self.staged_weights = None
        self.staged_tokens = None
        self.destroyed = True


def _validate_sm90_arch() -> None:
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major != 9:
        raise MoEEpArchError(
            "sm90_push_nvfp4 requires an SM90 (Hopper) device; "
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


@register_mega_kernel("sm90_push_nvfp4")
class Sm90PushNvFp4MegaKernelBackend(MegaKernelBackend):
    def __init__(self, config: Sm90PushNvFp4MegaMoeConfig) -> None:
        if not isinstance(config, Sm90PushNvFp4MegaMoeConfig):
            raise TypeError(
                "sm90_push_nvfp4 config must be Sm90PushNvFp4MegaMoeConfig, got "
                f"{type(config).__name__}"
            )
        super().__init__(config)
        self._kernel_config = config

    @classmethod
    def kernel_name(cls) -> str:
        return "sm90_push_nvfp4"

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
                "sm90_push_nvfp4 supports a single-node EP group of at most 32 ranks"
            )
        if config.top_k not in (1, 2, 4, 6, 8):
            raise MoEEpConfigError(
                "sm90_push_nvfp4 top_k must be one of (1, 2, 4, 6, 8)"
            )
        if config.intermediate_size <= 0 or config.intermediate_size % 128:
            raise MoEEpConfigError(
                "sm90_push_nvfp4 intermediate_size must be a positive multiple of 128"
            )
        if config.intermediate_size > 16384:
            raise MoEEpConfigError(
                "sm90_push_nvfp4 activation staging requires intermediate_size <= 16384"
            )
        if config.nvfp4_mode not in ("w4a8", "w4a16_rs"):
            raise MoEEpConfigError(
                "sm90_push_nvfp4 nvfp4_mode must be 'w4a8' or 'w4a16_rs'"
            )
        if config.weight_policy not in ("packed", "folded", "hot_folded", "dual"):
            raise MoEEpConfigError(
                "sm90_push_nvfp4 weight_policy must be packed, folded, "
                "hot_folded, or dual"
            )
        local_experts = fleet_params.num_experts // bootstrap.world_size
        if type(config.hot_expert_count) is not int or not (
            0 <= config.hot_expert_count <= local_experts
        ):
            raise MoEEpConfigError(
                "sm90_push_nvfp4 hot_expert_count must be in the local expert range"
            )
        if config.weight_policy == "packed" and config.hot_expert_count != 0:
            raise MoEEpConfigError("packed weight_policy requires hot_expert_count=0")
        if config.weight_policy == "folded" and config.hot_expert_count not in (
            0,
            local_experts,
        ):
            raise MoEEpConfigError(
                "folded weight_policy uses every local expert; hot_expert_count "
                "must be 0 or the local expert count"
            )
        if config.weight_policy == "hot_folded" and not (
            0 < config.hot_expert_count < local_experts
        ):
            raise MoEEpConfigError(
                "hot_folded weight_policy requires a nonempty proper hot prefix"
            )
        if config.weight_policy == "dual":
            if config.hot_expert_count != 0:
                raise MoEEpConfigError("dual weight_policy requires hot_expert_count=0")
            if config.acknowledge_dual_residency is not True:
                raise MoEEpConfigError(
                    "dual weight_policy retains both packed and folded weights "
                    "(577.6 MiB measured versus 241.5 MiB packed and 336.1 MiB "
                    "folded) and requires acknowledge_dual_residency=True"
                )
        if config.nvfp4_mode != "w4a8" and config.weight_policy != "packed":
            raise MoEEpConfigError(
                "non-packed weight policies require nvfp4_mode='w4a8'"
            )
        if type(config.tma_cache_capacity) is not int or not (
            1 <= config.tma_cache_capacity <= 128
        ):
            raise MoEEpConfigError(
                "sm90_push_nvfp4 tma_cache_capacity must be in [1, 128]"
            )
        try:
            n64_expected_m_per_sm = float(config.n64_expected_m_per_sm)
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(
                "sm90_push_nvfp4 n64_expected_m_per_sm must be finite and positive"
            ) from exc
        if not math.isfinite(n64_expected_m_per_sm) or n64_expected_m_per_sm <= 0.0:
            raise MoEEpConfigError(
                "sm90_push_nvfp4 n64_expected_m_per_sm must be finite and positive"
            )
        if config.payload_layout not in (3, 4):
            raise MoEEpConfigError("sm90_push_nvfp4 payload_layout must be 3 or 4")
        if type(config.allow_legacy_layout) is not bool:
            raise MoEEpConfigError("sm90_push_nvfp4 allow_legacy_layout must be a bool")
        if (
            config.nvfp4_mode == "w4a8"
            and config.payload_layout == 3
            and not config.allow_legacy_layout
        ):
            raise MoEEpConfigError(
                "sm90_push_nvfp4 payload_layout=3 is a legacy oracle and requires "
                "allow_legacy_layout=True"
            )
        if config.payload_dtype not in ("fp8", "bf16"):
            raise MoEEpConfigError(
                "sm90_push_nvfp4 payload_dtype must be 'fp8' or 'bf16'"
            )
        if config.combine_dtype not in ("fp8", "bf16"):
            raise MoEEpConfigError(
                "sm90_push_nvfp4 combine_dtype must be 'fp8' or 'bf16'"
            )
        if config.grouped_combine and config.combine_dtype != "fp8":
            raise MoEEpConfigError(
                "sm90_push_nvfp4 grouped_combine requires combine_dtype='fp8'"
            )
        if config.group_size not in (32, 64, 128):
            raise MoEEpConfigError("sm90_push_nvfp4 group_size must be 32, 64, or 128")
        if config.residual_scheme not in ("generic", "pow2"):
            raise MoEEpConfigError(
                "sm90_push_nvfp4 residual_scheme must be 'generic' or 'pow2'"
            )
        if config.nvfp4_mode == "w4a16_rs":
            if config.group_size != 128 or config.residual_scheme != "generic":
                raise MoEEpConfigError(
                    "sm90_push_nvfp4 w4a16_rs requires group_size=128 and "
                    "residual_scheme='generic'"
                )
            if config.combine_dtype != "bf16" or config.grouped_combine:
                raise MoEEpConfigError(
                    "sm90_push_nvfp4 w4a16_rs requires combine_dtype='bf16' "
                    "and grouped_combine=False"
                )
            if config.fuse_act:
                raise MoEEpConfigError(
                    "sm90_push_nvfp4 w4a16_rs requires fuse_act=False"
                )
            if config.rs_n_tactic != 64:
                raise MoEEpConfigError("sm90_push_nvfp4 rs_n_tactic must be 64")
            if config.rs_stages != 3:
                raise MoEEpConfigError("sm90_push_nvfp4 rs_stages must be 3")
            if config.rs_stage_k != 64:
                raise MoEEpConfigError("sm90_push_nvfp4 rs_stage_k must be 64")
            if (
                fleet_params.token_hidden_size % config.rs_stage_k
                or config.intermediate_size % config.rs_stage_k
            ):
                raise MoEEpConfigError(
                    "sm90_push_nvfp4 RS K dimensions must be divisible by rs_stage_k"
                )
        try:
            capacity_factor = float(config.capacity_factor)
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(
                "sm90_push_nvfp4 capacity_factor must be finite and in (0, 1]"
            ) from exc
        if not math.isfinite(capacity_factor) or not (0.0 < capacity_factor <= 1.0):
            raise MoEEpConfigError(
                "sm90_push_nvfp4 capacity_factor must be finite and in (0, 1]"
            )
        try:
            timeout_s = float(config.init_timeout_s)
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(
                "sm90_push_nvfp4 init_timeout_s must be finite and positive"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise MoEEpConfigError(
                "sm90_push_nvfp4 init_timeout_s must be finite and positive"
            )
        if bootstrap.stream != 0:
            raise MoEEpConfigError(
                "sm90_push_nvfp4 launches on the current torch CUDA stream; "
                "BootstrapConfig.stream must be 0"
            )

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> Any:
        config = self._kernel_config
        transformed = preprocess_mega_weights(
            weights,
            intermediate_size=config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            num_local_experts=fleet_params.num_experts // self.ep_world_size,
            nvfp4_mode=config.nvfp4_mode,
            group_size=config.group_size,
            residual_scheme=config.residual_scheme,
            payload_layout=config.payload_layout,
            weight_policy=config.weight_policy,
            hot_expert_count=config.hot_expert_count,
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
        config = self._kernel_config
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            num_local_experts=fleet_params.num_experts // self.ep_world_size,
            nvfp4_mode=config.nvfp4_mode,
            group_size=config.group_size,
            residual_scheme=config.residual_scheme,
            payload_layout=config.payload_layout,
            weight_policy=config.weight_policy,
            hot_expert_count=config.hot_expert_count,
        )
        self._transformed_weights = transformed_weights

    def _allocate_workspace(self, fleet_params: FleetParams) -> _Sm90PushNvFp4Workspace:
        from .....kernel_src.sm90.push_style_megamoe import (
            Sm90PushCombine,
            Sm90PushConfig,
            Sm90PushNvFp4MoERunner,
            Sm90PushPayload,
            Sm90PushPipe,
        )
        from ......comm.mnnvl import TorchDistBackend

        transformed_weights = self._transformed_weights
        if transformed_weights is None:
            raise RuntimeError(
                "sm90_push_nvfp4 weights must be prepared before workspace allocation"
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
                "sm90_push_nvfp4 failed to configure the EP process-group timeout: "
                + " | ".join(timeout_failures)
            )
        if any(peer_timeout != timeout_s for peer_timeout, _error in timeout_reports):
            raise RuntimeError(
                "sm90_push_nvfp4 init_timeout_s must match on every EP rank; got "
                f"{[peer_timeout for peer_timeout, _error in timeout_reports]}"
            )
        pipe = Sm90PushPipe(
            ep_size=self.ep_world_size,
            rank=self.ep_rank,
            num_local_experts=fleet_params.num_experts // self.ep_world_size,
            hidden_size=fleet_params.token_hidden_size,
            top_k=config.top_k,
            token_capacity=fleet_params.max_tokens_per_rank,
            device_index=torch.cuda.current_device(),
            config=Sm90PushConfig(
                payload_dtype=Sm90PushPayload(config.payload_dtype),
                combine_dtype=Sm90PushCombine(config.combine_dtype),
                fuse_act=config.fuse_act,
                capacity_factor=float(config.capacity_factor),
                dedup_dispatch=config.dedup_dispatch,
                grouped_combine=config.grouped_combine,
                fuse_fc1_epilogue=False,
            ),
            comm_backend=comm,
            out_dtype=torch.bfloat16,
            allow_unverified_p2p=config.allow_unverified_p2p,
        )
        try:
            runner = Sm90PushNvFp4MoERunner(
                pipe,
                transformed_weights,
                rs_n_tactic=config.rs_n_tactic,
                rs_stages=config.rs_stages,
                rs_stage_k=config.rs_stage_k,
                tma_cache_capacity=config.tma_cache_capacity,
                n64_expected_m_per_sm=float(config.n64_expected_m_per_sm),
                payload_layout=config.payload_layout,
                allow_legacy_layout=config.allow_legacy_layout,
            )
        except Exception:
            pipe.destroy()
            raise
        return _Sm90PushNvFp4Workspace(
            pipe=pipe,
            runner=runner,
            active_weights=transformed_weights,
        )

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        config = self._kernel_config
        transformed_weights = self._transformed_weights
        from .....kernel_src.sm90.push_style_megamoe import (
            Sm90PushNvFp4DualWeights,
            Sm90PushNvFp4HotFoldedWeights,
        )

        execution_identity = (
            transformed_weights.execution_identity
            if isinstance(
                transformed_weights,
                (Sm90PushNvFp4HotFoldedWeights, Sm90PushNvFp4DualWeights),
            )
            else (config.nvfp4_mode,)
        )
        return (
            "sm90_push_nvfp4",
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self.ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            fleet_params.token_hidden_size,
            config.intermediate_size,
            config.top_k,
            config.nvfp4_mode,
            config.weight_policy,
            config.hot_expert_count,
            config.acknowledge_dual_residency,
            execution_identity,
            config.group_size,
            config.residual_scheme,
            config.payload_dtype,
            config.combine_dtype,
            float(config.capacity_factor),
            config.dedup_dispatch,
            config.grouped_combine,
            config.fuse_act,
            config.rs_n_tactic,
            config.rs_stages,
            config.rs_stage_k,
            config.tma_cache_capacity,
            float(config.n64_expected_m_per_sm),
            config.payload_layout,
            config.allow_legacy_layout,
            config.allow_unverified_p2p,
            float(config.init_timeout_s),
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_sm90_push_nvfp4_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    @staticmethod
    def _live_workspace(workspace: Any) -> _Sm90PushNvFp4Workspace:
        if not isinstance(workspace, _Sm90PushNvFp4Workspace):
            raise TypeError("sm90_push_nvfp4 workspace must be created by this backend")
        if workspace.destroyed:
            raise RuntimeError("sm90_push_nvfp4 workspace has been destroyed")
        if workspace.poisoned:
            raise RuntimeError(
                "sm90_push_nvfp4 workspace is poisoned by an earlier failure"
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
                "sm90_push_nvfp4 requires MegaConfig.quantize_input=True"
            )
        ws = self._live_workspace(workspace)
        transformed_weights = self._transformed_weights
        if transformed_weights is None:
            raise RuntimeError(
                "sm90_push_nvfp4 weights must be preprocessed or validated before "
                "staging"
            )
        try:
            if ws.active_weights is not transformed_weights:
                ws.runner.bind_weights(transformed_weights)
                ws.active_weights = transformed_weights
            ws.runner.stage_inputs(
                t.hidden_states,
                t.topk_ids,
                t.topk_weights,
            )
        except Exception:
            if ws.runner.state == "poisoned":
                ws.poisoned = True
            raise
        ws.staged_weights = transformed_weights
        ws.staged_tokens = t.num_tokens

    def compute(
        self,
        workspace: Any,
        transformed_weights: Any,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        ws = self._live_workspace(workspace)
        staged_weights = ws.staged_weights
        num_tokens = ws.staged_tokens
        if num_tokens is None or staged_weights is None:
            raise RuntimeError(
                "sm90_push_nvfp4 compute() requires a successful stage_inputs()"
            )
        weights_mismatch = (
            transformed_weights is not staged_weights
            or self._transformed_weights is not staged_weights
        )
        try:
            result = ws.runner.compute(output=output)
        except Exception:
            if ws.runner.state == "poisoned":
                ws.poisoned = True
            raise
        finally:
            ws.staged_weights = None
            ws.staged_tokens = None
        if result is not output:
            raise RuntimeError(
                "sm90_push_nvfp4 runner must return the caller-provided output"
            )
        if weights_mismatch:
            raise RuntimeError(
                "sm90_push_nvfp4 compute received a different weight bundle; the "
                "staged round completed with its bound weights"
            )
        return output

    def destroy(self, workspace: Any) -> None:
        if workspace is None:
            return
        if not isinstance(workspace, _Sm90PushNvFp4Workspace):
            raise TypeError("sm90_push_nvfp4 workspace must be created by this backend")
        super().destroy(workspace)


__all__ = ["Sm90PushNvFp4MegaKernelBackend"]
