# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlashInfer MegaKernelBackend for the SM120 MXFP4 x MXFP8 nosync MegaMoE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.validation.common import (
    MoEEpArchError,
    MoEEpConfigError,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from ..mxfp4_mxfp8_bf16_cake.cake_staging import (
    quantize_activations_k32,
    validate_forward_inputs,
)
from .config import Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig
from .weights import preprocess_mega_weights, validate_transformed_mega_weights
from .workspace import NosyncWorkspace

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors

KERNEL_NAME = "sm120_mxfp4_mxfp8_bf16_nosync"
MAX_RANKS = 8
MAX_TOP_K = 8
MAX_LOCAL_EXPERTS = 256
# the dispatch stages one MXFP8 row (hidden + hidden / 32 bytes) in 8 KB of shared memory
MAX_HIDDEN = 7936
# A work item multiplies 8 to 32 tokens per weight fragment, a decode-size design. Up to 128
# tokens per rank it beats the sibling cutedsl and cake backends on RTX 6000D and RTX PRO 6000
# (DeepSeek-V4-Flash, DeepSeek-V4.1-Flash and GLM-5.3-Flash MoE shapes); larger batches belong on
# those backends.
MAX_TOKENS_PER_RANK = 128


@register_mega_kernel(KERNEL_NAME)
class Sm120Mxfp4Mxfp8NosyncMegaKernelBackend(MegaKernelBackend):
    supports_output_view = True

    def __init__(self, config: Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig) -> None:
        if not isinstance(config, Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig):
            raise TypeError(
                f"{KERNEL_NAME} config must be Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__(config)
        self._kernel_config = config
        if config.intermediate_size <= 0 or config.top_k <= 0:
            raise ValueError("intermediate_size and top_k must be positive")

    @classmethod
    def kernel_name(cls) -> str:
        return KERNEL_NAME

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # the module is built for sm_120a, which does not load on 12.1
            if capability != (12, 0):
                raise MoEEpArchError(
                    f"{KERNEL_NAME} requires sm_120 (compute capability 12.0); host has "
                    f"sm_{capability[0]}{capability[1]}"
                )
        config = self._kernel_config
        world = bootstrap.world_size
        validate_mega_fleet_params(
            fleet_params,
            world,
            intermediate_size=config.intermediate_size,
            top_k=config.top_k,
            alignment=256,
        )
        hidden = fleet_params.token_hidden_size
        checks = (
            (world <= MAX_RANKS, f"at most {MAX_RANKS} EP ranks (one host)"),
            (config.top_k <= MAX_TOP_K, f"top_k at most {MAX_TOP_K}"),
            (config.top_k <= fleet_params.num_experts, "top_k at most num_experts"),
            (
                fleet_params.num_experts // world <= MAX_LOCAL_EXPERTS,
                f"at most {MAX_LOCAL_EXPERTS} experts per rank",
            ),
            (hidden <= MAX_HIDDEN, f"token_hidden_size at most {MAX_HIDDEN}"),
            (
                0 < fleet_params.max_tokens_per_rank <= MAX_TOKENS_PER_RANK,
                f"max_tokens_per_rank in [1, {MAX_TOKENS_PER_RANK}] (a decode kernel; the "
                "sm120_mxfp4_mxfp8_bf16_cutedsl / _cake backends serve larger batches)",
            ),
        )
        for ok, requirement in checks:
            if not ok:
                raise MoEEpConfigError(f"{KERNEL_NAME} supports {requirement}")

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return preprocess_mega_weights(
            weights,
            hidden_size=fleet_params.token_hidden_size,
            intermediate_size=self._kernel_config.intermediate_size,
        )

    def validate_transformed_weights(
        self,
        transformed_weights: tuple[torch.Tensor, torch.Tensor],
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_transformed_mega_weights(
            transformed_weights,
            hidden_size=fleet_params.token_hidden_size,
            intermediate_size=self._kernel_config.intermediate_size,
            local_experts=fleet_params.num_experts // bootstrap.world_size,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> NosyncWorkspace:
        config = self._kernel_config
        return NosyncWorkspace(
            group=self._ep_comm_group,
            rank=self.ep_rank,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
            capacity=fleet_params.max_tokens_per_rank,
            top_k=config.top_k,
            hidden=fleet_params.token_hidden_size,
            intermediate=config.intermediate_size,
            transport=config.transport,
            fast_math=config.fast_math,
        )

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        config = self._kernel_config
        # tokens_per_item and swiglu_limit are launch arguments, not baked into the workspace
        return (
            KERNEL_NAME,
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            config.top_k,
            fleet_params.token_hidden_size,
            config.intermediate_size,
            config.transport,
            config.fast_math,
        )

    def validate_forward(
        self, t: "MoEEpTensors", fleet_params: FleetParams, *, quantize_input: bool
    ) -> None:
        validate_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    def set_compile_tokens_per_rank(
        self, workspace: Any, compile_tokens_per_rank: int | None
    ) -> None:
        """Rank-uniform bound on the next invocation's per-rank rows. The kernel takes each
        rank's row count as a launch argument and exchanges it on the device, so the bound
        is only checked; ranks may stage different row counts in the same call."""
        ws: NosyncWorkspace = workspace
        if compile_tokens_per_rank is not None and not (
            0 <= int(compile_tokens_per_rank) <= ws.capacity
        ):
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: compile_tokens_per_rank={compile_tokens_per_rank} must be in "
                f"[0, {ws.capacity}] (the workspace row capacity)"
            )
        ws.compile_tokens_per_rank = (
            None if compile_tokens_per_rank is None else int(compile_tokens_per_rank)
        )

    def stage_inputs(
        self, t: "MoEEpTensors", workspace: Any, *, quantize_input: bool
    ) -> None:
        ws: NosyncWorkspace = workspace
        rows = int(t.hidden_states.shape[0])
        bound = ws.compile_tokens_per_rank
        if bound is not None and rows > bound:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: compile_tokens_per_rank={bound} is below this rank's "
                f"{rows} tokens"
            )
        if quantize_input:
            codes, exponents = quantize_activations_k32(t.hidden_states)
        else:
            codes, exponents = (
                t.hidden_states.view(torch.uint8),
                t.scales.view(torch.uint8),
            )
        ws.x[:rows].copy_(codes)
        ws.x_sf[:rows].copy_(exponents)
        ws.topk_idx[:rows].copy_(t.topk_ids)
        ws.topk_weights[:rows].copy_(t.topk_weights)
        if rows == 0:  # the kernel runs one fully masked row for a rank without tokens
            ws.topk_idx[0].fill_(-1)
        ws.staged_rows = rows

    def workspace_output(self, workspace: Any) -> torch.Tensor:
        return workspace.output

    def compute(
        self,
        workspace: Any,
        transformed_weights: tuple[torch.Tensor, torch.Tensor],
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        ws: NosyncWorkspace = workspace
        rows = ws.staged_rows
        launch_rows = max(rows, 1)
        direct = (
            output is not None
            and output.dtype == torch.bfloat16
            and output.is_contiguous()
            and output.shape[0] >= launch_rows
            and output.shape[1] == ws.hidden
        )
        y = output if direct else ws.output
        w13, w2 = transformed_weights
        config = self._kernel_config
        ws.launch(y, w13, w2, launch_rows, config.swiglu_limit, config.tokens_per_item)
        if output is not None and not direct:
            output[:rows].copy_(ws.output[:rows])
            return output
        return y[:rows]

    def protocol_error(self, workspace: Any) -> int:
        """Error word of the last call (0 = every wait completed); synchronizes with the device."""
        return workspace.error()


__all__ = ["KERNEL_NAME", "Sm120Mxfp4Mxfp8NosyncMegaKernelBackend"]
