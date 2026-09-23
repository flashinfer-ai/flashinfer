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

    def _recv_count_for_exclusion(self, ctx: SplitKernelContext):
        """Per-expert REAL token counts, when they can be used to drop EP padding.

        Returning None is always safe: the bridge then marks every padded row as a
        real token, which is what this path did before - correct, just wasteful.

        EXPERT_MAJOR only. RANK_MAJOR's counts are per-source-rank rather than
        per-expert, and its rows are not laid out with a fixed per-expert stride, so
        the same tensor means something different there.

        **Opt-in**: set FLASHINFER_MOE_EP_EXCLUDE_PADDING_ROWS=1 to enable.

        Off by default because the win is strongly shape-dependent. Measured over 264
        A/B pairs on GB200 (4-32 ranks, NVFP4), end-to-end speedup ranges from 5.24x
        down to 0.84x, and the losing region grows with EP width: 31% of measured
        configurations regress at 4 ranks, 78% at 32. The cause is not an
        implementation artifact - profiling shows this path always reduces GPU work,
        but on small-MoE shapes the expert GEMM is only ~2.6% of the step (the rest
        being nccl_ep dispatch/combine), so there is not enough of it to pay for the
        transform.

        Rule of thumb for when to enable, from that data: turn it on when the
        baseline's per-GPU padded work clears roughly

            num_local_experts * cap * hidden_size * intermediate_size > 3e11

        which had zero regressions across the 240 configurations it selects.
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
                quant=quant,
                per_token_activation=per_token_activation,
                mxfp8_dispatch=self._mxfp8_dispatch,
                hidden_size=fleet_params.token_hidden_size,
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

        out_2d = self._ensure_compute(fleet_params)(act_pack, self._transformed_weights)
        return reshape_for_combine(out_2d, dim0, dim1)
