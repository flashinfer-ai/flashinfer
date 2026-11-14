"""Public moe_ep adapter for the isolated Kimi K3 kernel."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import TORCH_DIST
from ......core.validation.common import MoEEpArchError, MoEEpConfigError
from .config import Sm100_KimiK3_MegaMoeConfig

if TYPE_CHECKING:
    import torch

    from ......config import BootstrapConfig, FleetParams
    from ......tensors import MoEEpTensors
    from ......weights import MoEWeightPack


def _shim():
    from ......kernel_src.sm100 import _kimi_k3_megamoe

    return _kimi_k3_megamoe


@register_mega_kernel("sm100_kimi_k3_megamoe")
class KimiK3MegaKernelBackend(MegaKernelBackend):
    """Own Kimi workspaces through the common MegaMoE lifecycle."""

    _config: Sm100_KimiK3_MegaMoeConfig
    supports_output_view = True

    def __init__(self, config: Sm100_KimiK3_MegaMoeConfig) -> None:
        if not isinstance(config, Sm100_KimiK3_MegaMoeConfig):
            raise TypeError("Kimi K3 requires Sm100_KimiK3_MegaMoeConfig")
        super().__init__(replace(config, knobs=deepcopy(config.knobs)))

    @classmethod
    def kernel_name(cls) -> str:
        return "sm100_kimi_k3_megamoe"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return frozenset({TORCH_DIST}) if bootstrap.world_size > 1 else frozenset()

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        import torch

        for name, value, expected in (
            ("num_experts", fleet_params.num_experts, 896),
            ("token_hidden_size", fleet_params.token_hidden_size, 3584),
        ):
            if type(value) is not int or value != expected:
                raise MoEEpConfigError(f"Kimi K3 requires {name}={expected}")
        if type(bootstrap.world_size) is not int or bootstrap.world_size not in (
            1,
            8,
            16,
        ):
            raise MoEEpConfigError("Kimi K3 supports world_size 1, 8 or 16")
        if (
            type(fleet_params.max_tokens_per_rank) is not int
            or not 1 <= fleet_params.max_tokens_per_rank <= 65536
        ):
            raise MoEEpConfigError(
                "max_tokens_per_rank must be an integer in [1, 65536]"
            )
        if fleet_params.max_tokens_per_rank * self._config.top_k * 3584 >= 2**31:
            raise MoEEpConfigError(
                "max_tokens_per_rank exceeds the int32 combine-plane index limit"
            )
        if fleet_params.tuning_live_tokens is not None:
            raise MoEEpConfigError("Kimi K3 does not support tuning_live_tokens")
        if bootstrap.stream != 0:
            raise MoEEpConfigError("Kimi K3 uses the current CUDA stream; set stream=0")
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability not in ((10, 0), (10, 3)):
                raise MoEEpArchError("Kimi K3 requires SM100 or SM103")
        self._ensure_ep_bootstrap(bootstrap)

    def preprocess_weights(self, weights: MoEWeightPack, fleet_params: FleetParams):
        return _shim().preprocess_weights(
            weights, self._config, fleet_params, world_size=self.ep_world_size
        )

    def validate_transformed_weights(
        self,
        transformed_weights: Any,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        _shim().validate_transformed_weights(
            transformed_weights,
            self._config,
            fleet_params,
            world_size=bootstrap.world_size,
        )

    def _allocate_workspace(self, fleet_params: FleetParams):
        return _shim().create_workspace(
            self._config,
            fleet_params,
            rank=self.ep_rank,
            world_size=self.ep_world_size,
            process_group=self.ep_comm_group if self.ep_world_size > 1 else None,
        )

    def validate_forward(
        self,
        t: MoEEpTensors,
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        import torch

        x = t.hidden_states
        if x.ndim != 2:
            raise MoEEpConfigError("hidden_states must be a 2D tensor")
        tokens = x.shape[0]
        if tokens > fleet_params.max_tokens_per_rank:
            raise MoEEpConfigError("token count exceeds max_tokens_per_rank")
        nvfp4 = self._config.quant_kind == "nvfp4"
        hidden = fleet_params.token_hidden_size
        width = hidden if quantize_input or not nvfp4 else hidden // 2
        dtype = (
            torch.bfloat16
            if quantize_input
            else (torch.uint8 if nvfp4 else torch.float8_e4m3fn)
        )
        if x.shape[1] != width or x.dtype != dtype:
            raise MoEEpConfigError(
                f"hidden_states must have shape [tokens, {width}] and dtype {dtype}"
            )
        tensors = {
            "hidden_states": x,
            "topk_ids": t.topk_ids,
            "topk_weights": t.topk_weights,
        }
        if t.topk_ids.shape != (tokens, self._config.top_k):
            raise MoEEpConfigError("topk_ids must have shape [tokens, 16]")
        if t.topk_ids.dtype not in (torch.int32, torch.int64):
            raise MoEEpConfigError("topk_ids must have dtype int32 or int64")
        if t.topk_weights.shape != t.topk_ids.shape:
            raise MoEEpConfigError("topk_weights must have the same shape as topk_ids")
        if t.topk_weights.dtype != torch.float32:
            raise MoEEpConfigError("topk_weights must have dtype float32")
        if quantize_input:
            if t.scales is not None:
                raise MoEEpConfigError("scales must be omitted for BF16 input")
        else:
            if t.scales is None:
                raise MoEEpConfigError("prequantized input requires scales")
            scale_dtype = (
                torch.float8_e4m3fn
                if nvfp4
                else getattr(torch, "float8_e8m0fnu", torch.uint8)
            )
            if t.scales.shape != (
                tokens,
                hidden // (16 if nvfp4 else 32),
            ) or t.scales.dtype not in (torch.uint8, scale_dtype):
                raise MoEEpConfigError("scales must contain one byte per input block")
            tensors["scales"] = t.scales
        for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
            value = getattr(t, name)
            if value is None:
                continue
            if not nvfp4:
                raise MoEEpConfigError(
                    "MXFP4/MXFP8 does not accept alpha/norm overrides"
                )
            if value.dtype != torch.float32 or value.shape != (
                fleet_params.num_experts // self.ep_world_size,
            ):
                raise MoEEpConfigError(
                    f"{name} requires one FP32 value per local expert"
                )
            tensors[name] = value
        for name in (
            "fc1_activation_per_token_scale",
            "recv_count",
            "num_tokens_per_expert",
        ):
            if getattr(t, name) is not None:
                raise MoEEpConfigError(f"Kimi K3 does not support {name}")
        for name, value in tensors.items():
            if not value.is_cuda or value.device != x.device:
                raise MoEEpConfigError(f"{name} must be on the input CUDA device")
            if not value.is_contiguous():
                raise MoEEpConfigError(f"{name} must be contiguous")
            if name == "hidden_states" and quantize_input:
                alignment = 32
            elif name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
                alignment = 4
            else:
                alignment = 16
            if value.data_ptr() % alignment:
                raise MoEEpConfigError(f"{name} must be {alignment}-byte aligned")
        if x.device.index != torch.cuda.current_device():
            raise MoEEpConfigError("hidden_states must be on the current CUDA device")
        # The layer checks caller-output shape, dtype and device. Staging must
        # also establish Kimi's layout requirements before prepared execution.
        if t.output is not None and (
            not t.output.is_contiguous()
            or (t.output.numel() and t.output.data_ptr() % 16)
        ):
            raise MoEEpConfigError("Kimi output must be contiguous and 16-byte aligned")

    def stage_inputs(
        self, t: MoEEpTensors, workspace: Any, *, quantize_input: bool
    ) -> None:
        self.validate_forward(t, workspace.fleet_params, quantize_input=quantize_input)
        workspace.stage_inputs(t, quantize_input=quantize_input)

    def validate_capture_ready(self, workspace: Any, transformed_weights: Any) -> None:
        workspace.validate_capture_ready(transformed_weights)

    def workspace_output(self, workspace: Any) -> torch.Tensor:
        return workspace.output

    def compute(
        self, workspace: Any, transformed_weights: Any, *, output: torch.Tensor | None
    ) -> torch.Tensor:
        return workspace.compute(transformed_weights, output=output)

    def _compute_prepared(
        self, workspace: Any, transformed_weights: Any, *, output: torch.Tensor | None
    ) -> torch.Tensor:
        return workspace._compute(transformed_weights, output=output)
