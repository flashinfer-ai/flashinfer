"""Fused MoE split kernel — EP dispatch output through unified MoE compute."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

from .....config import BootstrapConfig, EpAlgorithm, EpLayout, FleetParams
from .....core.kernel.base import SplitKernelBackend, SplitKernelContext
from .....core.kernel.registry import register_split_kernel
from .....weights import MoEWeightPack
from .bridge import (
    build_activation_pack,
    build_activation_pack_rank_major,
    pack_mxfp8_dispatch_payload,
    reshape_for_combine,
)
from .config import FusedMoeKernelConfig
from .validate import validate_compute_consistency
from .weights import materialize_fused_moe_weights

if TYPE_CHECKING:
    from ......fused_moe.layer import MoELayer


@register_split_kernel("fused_moe")
class FusedMoeSplitKernelBackend(SplitKernelBackend):
    def __init__(self, config: FusedMoeKernelConfig) -> None:
        super().__init__(config)
        if not isinstance(config, FusedMoeKernelConfig):
            raise TypeError(
                f"FusedMoeSplitKernelBackend expects FusedMoeKernelConfig, "
                f"got {type(config).__name__}"
            )
        self._moe_config = config.moe_config
        self._mxfp8_dispatch = config.mxfp8_dispatch
        self._compute: Optional["MoELayer"] = None

    @classmethod
    def kernel_name(cls) -> str:
        return "fused_moe"

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_compute_consistency(fleet_params, bootstrap, self._moe_config)
        if self._mxfp8_dispatch:
            from ......fused_moe.api import CuteDslConfig, QuantVariant
            from .....core.validation.common import MoEEpConfigError

            if self._moe_config.quant.variant is not QuantVariant.MXFP4:
                raise MoEEpConfigError(
                    "mxfp8_dispatch requires MoEConfig quant variant MXFP4."
                )
            backends = tuple(self._moe_config.backend)
            if len(backends) != 1 or not isinstance(backends[0], CuteDslConfig):
                raise MoEEpConfigError(
                    "mxfp8_dispatch requires exactly one CuteDslConfig backend."
                )

    def pack_dispatch_payload(self, x):
        if not self._mxfp8_dispatch:
            return x
        return pack_mxfp8_dispatch_payload(x)

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ):
        self._transformed_weights = materialize_fused_moe_weights(
            weights, self._moe_config
        )
        return self._transformed_weights

    def _ensure_compute(self, fleet_params: FleetParams) -> "MoELayer":
        if self._compute is None:
            from ......fused_moe.layer import MoELayer

            cfg = self._moe_config
            received_routing = (
                fleet_params.layout is EpLayout.RANK_MAJOR
                or fleet_params.algorithm is EpAlgorithm.HIGH_THROUGHPUT
            )
            if received_routing:
                compute_cfg = cfg
            else:
                compute_cfg = dataclasses.replace(
                    cfg, routing=dataclasses.replace(cfg.routing, top_k=1)
                )
            self._compute = MoELayer(compute_cfg)
        return self._compute

    def compute(self, ctx: SplitKernelContext):
        expert_tensors = ctx.expert_tensors
        quant_variant = self._moe_config.quant.variant
        per_token_activation = bool(self._moe_config.quant.per_token_scale)
        offset = self._moe_config.experts.local_expert_offset
        dim0, dim1, _ = expert_tensors.shape

        fleet_params = ctx.fleet_params
        is_ht = fleet_params.algorithm is EpAlgorithm.HIGH_THROUGHPUT
        if is_ht or fleet_params.layout is EpLayout.RANK_MAJOR:
            if ctx.recv_topk_idx is None or ctx.recv_topk_weights is None:
                raise RuntimeError(
                    f"{'HT' if is_ht else 'RANK_MAJOR'} compute requires dispatch "
                    "to return recv_topk_idx / recv_topk_weights; got None."
                )
            act_pack = build_activation_pack_rank_major(
                expert_tensors,
                ctx.recv_topk_idx,
                ctx.recv_topk_weights,
                num_local_experts=self._moe_config.experts.local_num_experts,
                local_expert_offset=offset,
                quant_variant=quant_variant,
                per_token_activation=per_token_activation,
                mxfp8_dispatch=self._mxfp8_dispatch,
                hidden_size=fleet_params.token_hidden_size,
            )
        else:
            act_pack = build_activation_pack(
                expert_tensors,
                local_expert_offset=offset,
                quant_variant=quant_variant,
                per_token_activation=per_token_activation,
                mxfp8_dispatch=self._mxfp8_dispatch,
                hidden_size=fleet_params.token_hidden_size,
            )

        out_2d = self._ensure_compute(fleet_params)(act_pack, self._transformed_weights)
        return reshape_for_combine(out_2d, dim0, dim1)
