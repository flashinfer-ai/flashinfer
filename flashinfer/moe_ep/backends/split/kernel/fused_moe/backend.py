"""Fused MoE split kernel — EP dispatch output through unified MoE compute."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist

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
    from ...overlap import CombineInboxWorkspace


_TILE_SIGNAL_STATE_KEYS = (
    "tile_ready",
    "permuted_idx",
    "gemm2_c",
    "gemm2_mma_tiler_mn",
    "gemm2_cluster_shape_mn",
    "num_non_exiting_tiles",
    "permuted_m",
    "gemm2_ready_event",
)


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
        self._overlap_combine_fn = config.overlap_combine_fn
        self._compute: Optional["MoELayer"] = None
        self._overlap_enabled = False
        self._inbox: Optional["CombineInboxWorkspace"] = None
        self._consumer_stream: Optional[torch.cuda.Stream] = None
        self._topk_all: Optional[torch.Tensor] = None
        self._dest_fp: Optional[torch.Tensor] = None
        self._src_info: Optional[torch.Tensor] = None
        self._shipped_rows: Optional[torch.Tensor] = None
        self._src_rank = 0
        self.last_overlap_stats: dict[str, Any] = {}

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

    def _cute_dsl_flags(self) -> tuple[bool, bool]:
        from ......fused_moe.api import CuteDslConfig

        enable = False
        store_c = False
        for candidate in self._moe_config.backend:
            if isinstance(candidate, CuteDslConfig):
                enable = bool(candidate.enable_tile_signal)
                store_c = bool(candidate.store_permuted_c)
                break
        return enable, store_c

    def _overlap_wanted(self, fleet_params: FleetParams) -> bool:
        from ......fused_moe.api import QuantVariant

        enable, store_c = self._cute_dsl_flags()
        if not (enable and store_c):
            return False
        if self._moe_config.quant.variant is not QuantVariant.NVFP4:
            return False
        if fleet_params.algorithm is not EpAlgorithm.LOW_LATENCY:
            return False
        if fleet_params.layout is not EpLayout.EXPERT_MAJOR:
            return False
        return True

    def _cute_dsl_runner(self, layer: "MoELayer"):
        for runner in layer.runners:
            if getattr(runner, "backend_key", None) == "cute_dsl":
                return runner
        return None

    def _ensure_overlap(self, ctx: SplitKernelContext) -> None:
        from ...overlap import CombineInboxWorkspace, basic_overlap_combine

        fleet = ctx.fleet_params
        device = ctx.expert_tensors.device
        world = dist.get_world_size() if dist.is_initialized() else 1
        nle = int(
            self._moe_config.experts.local_num_experts
            or self._moe_config.routing.num_experts
        )
        tokens_per_rank = int(fleet.max_tokens_per_rank)
        hidden = int(fleet.token_hidden_size)
        if self._inbox is None:
            self._inbox = CombineInboxWorkspace(
                world_size=world,
                num_local_experts=nle,
                tokens_per_rank=tokens_per_rank,
                hidden=hidden,
                device=device,
            )
            self._consumer_stream = torch.cuda.Stream(device=device)
            dest_k = (
                int(ctx.dest_topk_ids.shape[1]) if ctx.dest_topk_ids is not None else 1
            )
            self._topk_all = torch.empty(
                (world, tokens_per_rank, dest_k),
                dtype=torch.int32,
                device=device,
            )
            self._dest_fp = torch.empty(
                (world, tokens_per_rank),
                dtype=torch.int64,
                device=device,
            )
            self._src_info = torch.full(
                (nle, world * tokens_per_rank),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self._shipped_rows = torch.zeros(1, dtype=torch.int32, device=device)
            self._src_rank = dist.get_rank() if dist.is_initialized() else 0
            if self._overlap_combine_fn is None:
                self._overlap_combine_fn = basic_overlap_combine
        assert self._inbox is not None and self._shipped_rows is not None
        self._inbox.zero()
        self._shipped_rows.zero_()

    def _arm_overlap(self, ctx: SplitKernelContext, layer: "MoELayer") -> None:
        runner = self._cute_dsl_runner(layer)
        if runner is None:
            raise RuntimeError(
                "fused-gemm2-combine requires the cute_dsl runner; "
                "none is configured on MoELayer."
            )
        if ctx.dest_topk_ids is None:
            raise RuntimeError(
                "overlap combine requires dest topk_ids on SplitKernelContext"
            )
        if ctx.dest_hidden_states is None:
            raise RuntimeError(
                "overlap combine requires dest hidden_states on SplitKernelContext"
            )
        self._ensure_overlap(ctx)
        fleet = ctx.fleet_params
        tokens_per_rank = int(fleet.max_tokens_per_rank)
        dest_ids = ctx.dest_topk_ids
        if dest_ids.shape[0] > tokens_per_rank:
            raise ValueError(
                f"dest topk_ids tokens={int(dest_ids.shape[0])} exceeds "
                f"max_tokens_per_rank={tokens_per_rank}"
            )
        local = torch.full(
            (tokens_per_rank, int(dest_ids.shape[1])),
            -1,
            dtype=torch.int32,
            device=dest_ids.device,
        )
        local[: dest_ids.shape[0]].copy_(dest_ids.to(dtype=torch.int32))
        world = int(self._topk_all.shape[0])
        topk_all_flat = self._topk_all.view(world * tokens_per_rank, -1)
        consumer = self._consumer_stream
        nle = int(
            self._moe_config.experts.local_num_experts
            or self._moe_config.routing.num_experts
        )
        offset = int(self._moe_config.experts.local_expert_offset)
        from ...overlap import (
            ROW_FP_UNUSED,
            combine_src_info_from_packed,
            row_fingerprint,
        )

        dest_h = ctx.dest_hidden_states
        n_tok = min(int(dest_h.shape[0]), tokens_per_rank)
        local_fp = torch.full(
            (tokens_per_rank,),
            ROW_FP_UNUSED,
            dtype=torch.int64,
            device=dest_h.device,
        )
        if n_tok > 0:
            local_fp[:n_tok] = row_fingerprint(dest_h[:n_tok])

        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer):
            if dist.is_initialized() and world > 1:
                dist.all_gather_into_tensor(topk_all_flat, local)
                dist.all_gather_into_tensor(self._dest_fp.view(-1), local_fp)
            else:
                self._topk_all[0].copy_(local)
                self._dest_fp[0].copy_(local_fp)
            combine_src_info_from_packed(
                ctx.expert_tensors,
                self._dest_fp,
                self._topk_all,
                local_expert_offset=offset,
                num_local_experts=nle,
                out=self._src_info,
            )

        state = runner.tile_signal_state
        state["on_gemm2_ready"] = self._on_gemm2_ready
        self._overlap_enabled = True

    def _on_gemm2_ready(self, state: dict[str, Any]) -> None:
        from ...overlap import launch_tile_ready_consumer

        missing = [
            k for k in _TILE_SIGNAL_STATE_KEYS if k not in state or state[k] is None
        ]
        if missing:
            raise RuntimeError(f"tile_signal_state missing keys: {missing}")
        inbox = self._inbox
        consumer = self._consumer_stream
        src_info = self._src_info
        shipped_rows = self._shipped_rows
        if (
            inbox is None
            or consumer is None
            or src_info is None
            or shipped_rows is None
        ):
            raise RuntimeError("overlap workspace was not armed in _arm_overlap")
        compile_only = bool(state.get("consumer_compile_only"))
        if not compile_only:
            consumer.wait_event(state["gemm2_ready_event"])
        nle = int(
            self._moe_config.experts.local_num_experts
            or self._moe_config.routing.num_experts
        )
        with torch.cuda.stream(consumer):
            if not compile_only:
                shipped_rows.zero_()
            launch_tile_ready_consumer(
                tile_ready=state["tile_ready"],
                gemm2_c=state["gemm2_c"],
                permuted_idx_to_expanded_idx=state["permuted_idx"],
                mma_tiler_mn=state["gemm2_mma_tiler_mn"],
                tokens_per_rank=int(inbox.tokens_per_rank),
                world_size=int(inbox.world_size),
                num_local_experts=nle,
                num_non_exiting_tiles=state["num_non_exiting_tiles"],
                permuted_m=int(state["permuted_m"]),
                src_info=src_info,
                peer_ptrs=inbox.peer_ptrs,
                src_rank=self._src_rank,
                shipped_rows=shipped_rows,
                stream=consumer,
                compile_only=compile_only,
            )

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
        layer = self._ensure_compute(fleet_params)
        if self._overlap_wanted(fleet_params):
            self._arm_overlap(ctx, layer)
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

        out_2d = layer(act_pack, self._transformed_weights)
        return reshape_for_combine(out_2d, dim0, dim1)

    def wait_overlap(self) -> None:
        if not self._overlap_enabled or self._consumer_stream is None:
            return
        torch.cuda.current_stream().wait_stream(self._consumer_stream)

    def collect_overlap_combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._inbox is None or self._overlap_combine_fn is None:
            raise RuntimeError("overlap combine was not armed for this iteration")
        hidden = int(hidden_states.shape[-1])
        live_rows = (
            int(self._shipped_rows.item()) if self._shipped_rows is not None else 0
        )
        live_bytes = live_rows * hidden * 2
        expected = 0
        if self._topk_all is not None:
            offset = int(self._moe_config.experts.local_expert_offset)
            nle = int(
                self._moe_config.experts.local_num_experts
                or self._moe_config.routing.num_experts
            )
            expected = int(
                ((self._topk_all >= offset) & (self._topk_all < offset + nle))
                .sum()
                .item()
            )
        pack_rows = (
            int(self._inbox.num_local_experts)
            * int(self._inbox.tokens_per_rank)
            * int(self._inbox.world_size)
        )
        pack_bytes = pack_rows * hidden * 2
        self.last_overlap_stats = {
            "live_rows": live_rows,
            "live_bytes": live_bytes,
            "expected_rows": expected,
            "pack_bytes": pack_bytes,
        }
        # Abort only if the consumer shipped the padded pack, or many more
        # rows than dest-local top-k hits. Do not use pack/4: occupancy is
        # topk/num_experts (50% on the 16-expert correctness geometry).
        # live != expected is stats-only: autotune/graph can skip or replay
        # the consumer. The pytest path asserts equality separately.
        if pack_rows > 0 and live_rows >= pack_rows:
            raise RuntimeError(
                f"overlap ship live_rows={live_rows} is the full padded pack "
                f"(pack_rows={pack_rows}); sparse live filter is not working"
            )
        if expected > 0 and live_rows > 2 * expected:
            raise RuntimeError(
                f"overlap ship live_rows={live_rows} >> expected_rows={expected} "
                f"(pack_rows={pack_rows})"
            )
        return self._overlap_combine_fn(
            self._inbox, hidden_states, topk_ids, topk_weights
        )

    def destroy(self) -> None:
        if self._inbox is not None:
            self._inbox.destroy()
            self._inbox = None
        self._consumer_stream = None
        self._topk_all = None
        self._dest_fp = None
        self._src_info = None
        self._shipped_rows = None
        self._overlap_enabled = False
        self._compute = None
