"""SM90 native BF16 (Cake-generated GEMMs on the push protocol) mega-MoE kernel backend."""

from __future__ import annotations

import math
import warnings
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
from .cake_config import Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig
from .cake_staging import validate_sm90_cake_bf16_forward_inputs
from .cake_weights import (
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors

_NAME = "sm90_bf16_bf16_bf16_push_cake"


@dataclass
class _Sm90CakeBf16Workspace:
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
            f"{_NAME} requires an SM90 (Hopper) device; host has sm_{major}{minor}"
        )


def _process_group_backend_timeouts(group: Any) -> list[tuple[Any, timedelta]]:
    """Best-effort snapshot of the per-backend default timeouts of ``group``.

    Mirrors the backend enumeration of
    ``torch.distributed.distributed_c10d._set_pg_timeout`` so that the init
    timeout can be undone afterwards.  Returns an empty list when this PyTorch
    build exposes no timeout getter; the caller then leaves ``init_timeout_s``
    in place (see ``Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig``).
    """
    import torch.distributed as dist

    if group is None:
        get_default_group = getattr(
            getattr(dist, "distributed_c10d", None), "_get_default_group", None
        )
        if get_default_group is None:
            return []
        group = get_default_group()
    try:
        devices = list(group._device_types)
    except Exception:  # noqa: BLE001 - private API, absent on some builds
        return []
    snapshot: list[tuple[Any, timedelta]] = []
    for device in devices:
        try:
            backend = group._get_backend(device)
            timeout = backend.options._timeout
        except Exception:  # noqa: BLE001 - private API, absent on some builds
            continue
        if isinstance(timeout, timedelta):
            snapshot.append((backend, timeout))
    return snapshot


def _restore_process_group_timeouts(snapshot: list[tuple[Any, timedelta]]) -> None:
    for backend, timeout in snapshot:
        try:
            backend._set_default_timeout(timeout)
        except Exception as exc:  # noqa: BLE001 - never mask the workspace result
            warnings.warn(
                f"{_NAME} could not restore the EP process-group timeout "
                f"{timeout}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
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
            "torch.distributed exposes neither set_timeout nor distributed_c10d._set_pg_timeout"
        )
    set_pg_timeout(timeout, group)


@register_mega_kernel(_NAME)
class Sm90CakeBf16MegaKernelBackend(MegaKernelBackend):
    """``MegaKernelBackend`` for native BF16 expert compute on Hopper.

    Dispatch/combine reuse the SM90 push protocol (bf16 payload, bf16 combine
    wire); FC1 (fused SwiGLU) and FC2 are Cake-generated WGMMA grouped GEMMs with
    bf16 operands and fp32 accumulation.
    """

    def __init__(self, config: Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig) -> None:
        if not isinstance(config, Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig):
            raise TypeError(
                f"{_NAME} config must be Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig, got "
                f"{type(config).__name__}"
            )
        super().__init__(config)
        self._kernel_config = config
        self._transformed_weights: Any | None = None

    @classmethod
    def kernel_name(cls) -> str:
        return _NAME

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        del bootstrap
        return frozenset({TORCH_DIST})

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
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
                f"{_NAME} supports a single-node EP group of at most 32 ranks, got "
                f"world_size={bootstrap.world_size}"
            )
        if kcfg.top_k not in (1, 2, 4, 6, 8):
            raise MoEEpConfigError(
                f"{_NAME} top_k must be one of (1, 2, 4, 6, 8), got {kcfg.top_k}"
            )
        hidden = fleet_params.token_hidden_size
        inter = kcfg.intermediate_size
        if hidden % 256 != 0 or inter % 128 != 0:
            raise MoEEpConfigError(
                f"{_NAME} requires token_hidden_size % 256 == 0 and intermediate_size % 128 == 0 "
                f"(WGMMA 256-column tiles and the gate/up interleave), got hidden={hidden}, "
                f"intermediate={inter}"
            )
        if fleet_params.num_experts % bootstrap.world_size != 0:
            raise MoEEpConfigError(
                f"{_NAME} requires num_experts ({fleet_params.num_experts}) to be divisible by "
                f"world_size ({bootstrap.world_size})"
            )
        try:
            capacity_factor = float(kcfg.capacity_factor)
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(
                f"{_NAME} capacity_factor must be finite and in (0, 1]"
            ) from exc
        if not math.isfinite(capacity_factor) or not (0.0 < capacity_factor <= 1.0):
            raise MoEEpConfigError(
                f"{_NAME} capacity_factor must be finite and in (0, 1], got {kcfg.capacity_factor!r}"
            )
        if kcfg.clamp_limit is not None:
            try:
                clamp = float(kcfg.clamp_limit)
            except (TypeError, ValueError) as exc:
                raise MoEEpConfigError(
                    f"{_NAME} clamp_limit must be a positive finite float or None"
                ) from exc
            if not math.isfinite(clamp) or clamp <= 0.0:
                raise MoEEpConfigError(
                    f"{_NAME} clamp_limit must be a positive finite float or None, got {kcfg.clamp_limit!r}"
                )
        try:
            timeout_s = float(kcfg.init_timeout_s)
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(
                f"{_NAME} init_timeout_s must be finite and positive"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise MoEEpConfigError(
                f"{_NAME} init_timeout_s must be finite and positive, got {kcfg.init_timeout_s!r}"
            )
        if bootstrap.stream != 0:
            raise MoEEpConfigError(
                f"{_NAME} launches on the current torch CUDA stream; BootstrapConfig.stream must be 0"
            )

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> Any:
        transformed = preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            num_local_experts=fleet_params.num_experts // self.ep_world_size,
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
        )
        self._transformed_weights = transformed_weights

    def _allocate_workspace(self, fleet_params: FleetParams) -> _Sm90CakeBf16Workspace:
        from .......comm.mnnvl import TorchDistBackend

        transformed_weights = self._transformed_weights
        if transformed_weights is None:
            raise RuntimeError(
                f"{_NAME} weights must be preprocessed or validated before workspace allocation"
            )
        comm = TorchDistBackend(group=self.ep_comm_group)
        # The caller's EP group timeout is raised to init_timeout_s only for
        # the setup below (JIT builds, handle exchange) and restored on exit.
        previous_timeouts = _process_group_backend_timeouts(self.ep_comm_group)
        try:
            return self._allocate_workspace_with_init_timeout(
                fleet_params, comm, transformed_weights
            )
        finally:
            _restore_process_group_timeouts(previous_timeouts)

    def _allocate_workspace_with_init_timeout(
        self, fleet_params: FleetParams, comm: Any, transformed_weights: Any
    ) -> _Sm90CakeBf16Workspace:
        from ......kernel_src.sm90.cake_bf16_megamoe import Sm90CakeBf16MoERunner
        from ......kernel_src.sm90.push_style_megamoe import (
            Sm90PushCombine,
            Sm90PushConfig,
            Sm90PushPayload,
            Sm90PushPipe,
        )

        kcfg = self._kernel_config
        timeout_s = float(kcfg.init_timeout_s)
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
                f"{_NAME} failed to configure the EP process-group timeout: "
                + " | ".join(timeout_failures)
            )
        if any(peer_timeout != timeout_s for peer_timeout, _error in timeout_reports):
            raise RuntimeError(
                f"{_NAME} init_timeout_s must match on every EP rank; got "
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
                payload_dtype=Sm90PushPayload.BF16,
                combine_dtype=Sm90PushCombine.BF16,
                fuse_act=True,
                capacity_factor=float(kcfg.capacity_factor),
                dedup_dispatch=kcfg.dedup_dispatch,
                grouped_combine=False,
                fuse_fc1_epilogue=False,
            ),
            comm_backend=comm,
            out_dtype=torch.bfloat16,
            allow_unverified_p2p=kcfg.allow_unverified_p2p,
        )
        try:
            runner = Sm90CakeBf16MoERunner(
                pipe,
                transformed_weights,
                clamp=None if kcfg.clamp_limit is None else float(kcfg.clamp_limit),
            )
        except Exception:
            pipe.destroy()
            raise
        return _Sm90CakeBf16Workspace(
            pipe=pipe, runner=runner, active_weights=transformed_weights
        )

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        kcfg = self._kernel_config
        return (
            _NAME,
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self.ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            fleet_params.token_hidden_size,
            kcfg.intermediate_size,
            kcfg.top_k,
            float(kcfg.capacity_factor),
            kcfg.dedup_dispatch,
            None if kcfg.clamp_limit is None else float(kcfg.clamp_limit),
            kcfg.allow_unverified_p2p,
            float(kcfg.init_timeout_s),
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_sm90_cake_bf16_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    @staticmethod
    def _live_workspace(workspace: Any) -> _Sm90CakeBf16Workspace:
        if not isinstance(workspace, _Sm90CakeBf16Workspace):
            raise TypeError(
                f"{_NAME} workspace must be created by this backend, got {type(workspace).__name__}"
            )
        if workspace.destroyed:
            raise RuntimeError(f"{_NAME} workspace has been destroyed")
        if workspace.poisoned:
            raise RuntimeError(
                f"{_NAME} workspace is poisoned by an earlier round failure"
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
                f"{_NAME} requires MegaConfig.quantize_input=True (native bf16 inputs)"
            )
        ws = self._live_workspace(workspace)
        transformed_weights = self._transformed_weights
        if transformed_weights is None:
            raise RuntimeError(
                f"{_NAME} weights must be preprocessed or validated before staging"
            )
        try:
            if ws.active_weights is not transformed_weights:
                ws.runner.bind_weights(transformed_weights)
                ws.active_weights = transformed_weights
            ws.runner.stage_inputs(t.hidden_states, t.topk_ids, t.topk_weights)
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
                f"{_NAME} compute() requires a successful stage_inputs()"
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
                f"{_NAME} runner must return the caller-provided output tensor"
            )
        if weights_mismatch:
            raise RuntimeError(
                f"{_NAME} compute received a different weight bundle; the staged round completed "
                "with its bound weights"
            )
        return output

    def destroy(self, workspace: Any) -> None:
        if workspace is None:
            return
        if not isinstance(workspace, _Sm90CakeBf16Workspace):
            raise TypeError(
                f"{_NAME} workspace must be created by this backend, got {type(workspace).__name__}"
            )
        super().destroy(workspace)


__all__ = ["Sm90CakeBf16MegaKernelBackend"]
