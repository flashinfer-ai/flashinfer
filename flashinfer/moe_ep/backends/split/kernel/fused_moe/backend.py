"""Fused MoE split kernel — EP dispatch output through unified MoE compute."""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Optional

from .....config import BootstrapConfig, EpAlgorithm, EpLayout, FleetParams
from .....core.kernel.base import SplitKernelBackend, SplitKernelContext
from .....core.kernel.registry import register_split_kernel
from .....weights import MoEWeightPack
from .bridge import (
    build_activation_pack,
    build_activation_pack_rank_major,
    dispatch_quant_format,
    pack_mxfp8_dispatch_payload,
    quantize_for_dispatch,
    reshape_for_combine,
)
from .config import FusedMoeKernelConfig
from .validate import validate_compute_consistency
from .weights import materialize_fused_moe_weights

if TYPE_CHECKING:
    from ......fused_moe.api import RoutingInputMode
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
        self._global_routing_mode: Optional["RoutingInputMode"] = None

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
            from ......fused_moe.api import CuteDslConfig, QuantFormat
            from .....core.validation.common import MoEEpConfigError

            if self._moe_config.quant.pair != (
                QuantFormat.MXFP4,
                QuantFormat.MXFP8,
            ):
                raise MoEEpConfigError(
                    "mxfp8_dispatch requires MoEConfig quant pair MXFP4×MXFP8."
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

    @property
    def packs_dispatch_payload(self) -> bool:
        return self._mxfp8_dispatch

    def dispatch_format(self):
        quant = self._moe_config.quant
        return dispatch_quant_format(
            quant,
            per_token_activation=bool(quant.per_token_scale),
            mxfp8_dispatch=self._mxfp8_dispatch,
        )

    def quantize_for_dispatch(self, x):
        return quantize_for_dispatch(x, self._moe_config.quant)

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

    def _global_ids_routing_mode(self, compute: "MoELayer") -> "RoutingInputMode":
        """Pre-routed mode for received global ids: unpacked when every runner
        takes it, so the ids and weights reach the routing kernel as they are."""
        if self._global_routing_mode is None:
            from ......fused_moe.api import RoutingInputMode

            unpacked = RoutingInputMode.UnpackedPrecomputed
            self._global_routing_mode = (
                unpacked
                if all(unpacked in r.supported_routing_modes for r in compute.runners)
                else RoutingInputMode.PackedPrecomputed
            )
        return self._global_routing_mode

    def _recv_count_for_exclusion(self, ctx: SplitKernelContext):
        """Per-expert REAL token counts, to be used to drop EP padding.

        Guarded by env var FLASHINFER_MOE_EP_EXCLUDE_PADDING_ROWS=1 (off by default)

        Returns None if all the rows should be considered real tokens
        """
        if ctx.recv_count is None:
            return None
        if ctx.fleet_params.layout is not EpLayout.EXPERT_MAJOR:
            return None
        if os.environ.get("FLASHINFER_MOE_EP_EXCLUDE_PADDING_ROWS", "0") != "1":
            return None
        return ctx.recv_count

    def compute(self, ctx: SplitKernelContext):
        expert_tensors = ctx.expert_tensors
        quant = self._moe_config.quant
        per_token_activation = bool(quant.per_token_scale)
        offset = self._moe_config.experts.local_expert_offset
        dim0, dim1, _ = expert_tensors.shape

        fleet_params = ctx.fleet_params
        compute = self._ensure_compute(fleet_params)
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
                num_experts=self._moe_config.routing.num_experts,
                quant=quant,
                per_token_activation=per_token_activation,
                mxfp8_dispatch=self._mxfp8_dispatch,
                hidden_size=fleet_params.token_hidden_size,
                recv_scale=ctx.expert_tensors_scale,
                global_ids=ctx.recv_topk_idx_global,
                routing_input_mode=(
                    self._global_ids_routing_mode(compute)
                    if ctx.recv_topk_idx_global
                    else None
                ),
            )
        else:
            act_pack = build_activation_pack(
                expert_tensors,
                local_expert_offset=offset,
                quant=quant,
                per_token_activation=per_token_activation,
                mxfp8_dispatch=self._mxfp8_dispatch,
                hidden_size=fleet_params.token_hidden_size,
                recv_count=self._recv_count_for_exclusion(ctx),
                num_experts=self._moe_config.routing.num_experts,
            )

        out_2d = compute(act_pack, self._transformed_weights)
        return reshape_for_combine(out_2d, dim0, dim1)

    def destroy(self) -> None:
        # The runners keep the last routing tensors they were given (to seed
        # tuning inputs), which are views of the transport's receive buffers.
        self._compute = None
        self._global_routing_mode = None
