# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MegaMoE fused dispatch + fc1 + fc2 + combine kernel (FP8).

Parallel to the NVFP4 swap-AB MegaMoE kernel and shared by the SM90 FP8
non-swap and swap-AB fused fc1+fc2 bases.
The token-communication machinery (dispatch prep / barrier / pull, NVLink
barrier, kernel tail) is reused verbatim from
``src/token_comm.py`` via the ``TokenInPullTokenBackPush`` helper; only the
data-format-dependent workspace layout differs:

  - ``hidden_bytes = hidden``                 (fp8 = 1 byte/element vs NVFP4 /2)
  - ``fc1_output`` region dtype = ``ab_dtype``  (Float8E4M3FN or Float8E5M2)
  - ``fc1_output_sf`` region dtype = ``Float8E8M0FNU`` for per-tensor,
    ``Float32`` for blockwise FC2 activation scale
  - dispatch scale atom covers 128 K elements: either four legacy E8M0 SF
    bytes or one blockwise FP32 scale; blockwise rows are padded to a 16-byte
    stride for TMA
  - ``Fp8GateUpInterleave = 8``                (FC1 gate/up layout)

The caller only provides the final ``output_activation`` with shape
``(max_tokens_per_rank, hidden)``.  The pre-topk-reduce plane is internal.
Following the NVFP4 mega kernel, the combine surface is two orthogonal knobs:

``token_back_mode`` -- who performs the cross-rank fc2 write-back:

  * ``epi_warps``: the fc2 epilogue STGs straight to the source rank.
  * ``reuse_dispatch_warps``: the epilogue stages rows in the local
    ``fc2_output_workspace`` pool; dispatch warps bulk-push them after pull.
  * ``standalone_warps``: same staging, pushed by four dedicated warps.

``fc2_in_kernel_topk_reduce`` -- where the topk axis collapses:

  * False (**separate reduce**): writes land in the internal
    ``combine_quant[src_token, src_topk, :]`` shared-symmetric staging, then
    the shared ``TopkReduce`` kernel collapses topk into the public output.
  * True (**in-kernel reduce**): writes accumulate directly into a
    ``(max_tokens_per_rank, 1, hidden)`` view of the public output --
    ``epi_warps`` issues ``red.relaxed.sys.global.add.noftz.v2.bf16x2``, the
    dispatch modes push with ``cp.reduce.async.bulk...add.noftz.bf16``
    (``token_back_reduce_topk``); no reducer kernel runs.

``static_expert_shape`` is required because dispatch storage and pool sizes are
codegen-time quantities.
"""

# NOTE: ``from __future__ import annotations`` is intentionally NOT used here
# (PEP 563 string-ifies class-body annotations, which breaks ``@cute.struct``
# element-type introspection).  See moe_nvfp4_swapab/megamoe_kernel.py.

from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import (
    Int64,
    Int32,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass.base_dsl.dsl import extract_mlir_attributes
from cutlass._mlir import ir

from common.host_utils import get_cutedsl_target_arch

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from src.iket_compat import iket

from moe_hopper_fp8.kernel_fp8_glu_fc12 import (
    Sm90SwigluFp8Fc12Kernel,
)
from moe_hopper_fp8.kernel_fp8_glu_fc12_swapab import (
    Sm90SwapABSwigluFp8Fc12Kernel,
)
from moe_hopper_fp8.kernel_mxfp4_fp8_glu_fc12_swapab import (
    Sm90SwapABSwigluMxfp4Fp8Fc12Kernel,
)
from moe_nvfp4_swapab.moe_utils import spin_wait
from moe_nvfp4_swapab.topk_reduce import TopkReduce
from src.token_comm import (
    CombineFormat,
    TokenCommArgs as ExtractedTokenCommArgs,
    TokenInPullTokenBackPush,
)
from src.sym_buffer import SingleRankSymBufferDevice
from common.megamoe_constants import (
    Fp8DispatchScaleAtomK,
    Fp8E8M0SfVecSize,
    Fp8Fc2ActivationScaleK,
)

# Reuse the region-layout helpers + module constants from the NVFP4 mega kernel
# so the two paths stay byte-for-byte consistent in their workspace plumbing.
from moe_nvfp4_swapab.megamoe_kernel import (
    _RegionSpec,
    _round_up,
    _layout_regions,
    _DispatchWarpCount,
    _TokenMetadataBytes,
    _GridSyncSlotCount,
    _NvlinkSlotCount,
)


# =============================================================================
# Split-role TokenComm ABIs
# =============================================================================


class _SplitTokenCommArgs:
    """Concrete JIT-serializable argument bundle for one split role."""

    _mlir_value_fields: Tuple[str, ...] = ()
    _const_fields: Tuple[str, ...] = ()

    def __init__(self, **kwargs) -> None:
        expected = set(self._mlir_value_fields + self._const_fields)
        provided = set(kwargs)
        if provided != expected:
            raise TypeError(
                f"{type(self).__name__} fields mismatch: "
                f"missing={sorted(expected - provided)} "
                f"unexpected={sorted(provided - expected)}"
            )
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for name in self._mlir_value_fields:
            values.extend(extract_mlir_values(getattr(self, name)))
        return values

    def __extract_mlir_attributes__(self) -> List[Any]:
        attrs: List[Any] = []
        for name in self._mlir_value_fields:
            attrs.extend(extract_mlir_attributes(getattr(self, name)))
        return attrs

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "_SplitTokenCommArgs":
        index = 0
        rebuilt: Dict[str, Any] = {}
        for name in self._mlir_value_fields:
            prototype = getattr(self, name)
            value_count = len(extract_mlir_values(prototype))
            rebuilt[name] = new_from_mlir_values(
                prototype, values[index : index + value_count]
            )
            index += value_count
        assert index == len(values), (
            f"{type(self).__name__} serialization mismatch: "
            f"consumed={index} provided={len(values)}"
        )
        for name in self._const_fields:
            rebuilt[name] = getattr(self, name)
        return type(self)(**rebuilt)


class SplitK1TokenCommArgs(_SplitTokenCommArgs):
    """Dispatch/FC1-only communication state; no combine/K2 fields."""

    _mlir_value_fields = (
        "input_token_buffer",
        "input_sf_buffer",
        "topk_idx",
        "input_topk_weights_buffer",
        "expert_send_count",
        "expert_recv_count",
        "expert_recv_count_sum",
        "src_token_topk_idx",
        "fc1_input_token_buffer",
        "fc1_input_sf_buffer",
        "fc1_input_topk_weights_buffer",
        "fc1_ready_counter",
        "token_src_metadata",
        "nvlink_barrier_signal",
        "nvlink_barrier_counter",
        "grid_sync_counter",
        "local_zero_prefix",
        "shared_zero_prefix",
        "split_dispatch_ready",
        "peer_rank_ptr_mapper",
        "local_rank",
    )
    _const_fields = (
        "world_size",
        "num_total_experts",
        "num_experts_per_rank",
        "num_topk",
        "hidden_bytes",
        "sf_uint32_per_token",
        "token_padding_block",
        "sf_padding_block",
        "sm_count",
    )


class SplitK2TokenCommArgs(_SplitTokenCommArgs):
    """FC2/direct-combine state; no dispatch or FC1-ready fields."""

    _mlir_value_fields = (
        "token_src_metadata",
        "combine_output",
        "split_dispatch_ready",
        "peer_rank_ptr_mapper",
    )


# =============================================================================
# Sm90MegaMoEFp8Kernel
# =============================================================================


class Sm90MegaMoEFp8Kernel(Sm90SwigluFp8Fc12Kernel):
    """MegaMoE-complete fused dispatch + fc1 + fc2 + combine kernel."""

    def __init__(
        self,
        # Base-class kwargs (forwarded 1:1 to ``super().__init__``).
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        load_balance_mode: str = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        ab_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        sf_vec_size: int = Fp8E8M0SfVecSize,
        fp8_scale_mode: str = "per_tensor",
        fp8_accum_mode: str = "1xacc",
        execution_phase: Literal["fused", "fc1", "fc2"] = "fused",
        pingpong: bool = False,
        scenario: str = "2Dx3D",
        # MegaMoE-specific independent constants.
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        max_tokens_per_rank: int,
        hidden: int,
        fc2_in_kernel_topk_reduce: bool = False,
        apply_topk_in_fc1: bool = True,
        token_back_mode: Literal[
            "epi_warps", "standalone_warps", "reuse_dispatch_warps"
        ] = "epi_warps",
        epi_flag_batch: Union[int, Tuple[int, int]] = 1,
        flag_batch: int = 1,
        gate_up_clamp: Optional[float] = None,
        split_role: str = "fused",
        split_fc1_tile_m: Optional[int] = None,
        split_fc1_token_n: Optional[int] = None,
        split_handoff_token_n: Optional[int] = None,
        split_workspace_counter_tile_tokens: Optional[int] = None,
        split_counter_epoch_banks: int = 1,
        split_counter_epoch_bank: int = 0,
    ) -> None:
        if token_back_mode not in (
            "epi_warps", "standalone_warps", "reuse_dispatch_warps",
        ):
            raise ValueError(f"unsupported token_back_mode '{token_back_mode}'")
        # PR4688 keeps the explicit three-mode frontend while the donor split
        # contract names the same device choice as a boolean.
        token_back_by_dispatch = token_back_mode != "epi_warps"
        if split_role not in ("fused", "k1", "k2"):
            raise ValueError(
                "split_role must be 'fused', 'k1', or 'k2'; "
                f"got {split_role!r}."
            )
        if split_role != "fused" and pingpong:
            raise ValueError(
                "split K1/K2 execution currently requires pingpong=False."
            )
        if split_role != "fused" and token_back_by_dispatch:
            raise ValueError(
                "split K1/K2 use direct combine and cannot enable "
                "token_back_by_dispatch."
            )
        if split_role != "fused" and fc2_in_kernel_topk_reduce:
            raise ValueError(
                "split K1/K2 require standalone K3 TopkReduce."
            )
        if split_role == "k2" and split_fc1_tile_m not in (128, 256):
            raise ValueError(
                "split K2 requires split_fc1_tile_m=128 or 256."
            )
        if split_role != "fused" and split_fc1_token_n not in (16, 32, 64, 128):
            raise ValueError(
                "split K1/K2 require split_fc1_token_n in "
                "(16, 32, 64, 128)."
            )
        if split_handoff_token_n is not None:
            if split_role == "fused":
                raise ValueError("fused MegaMoE cannot use a split handoff tile.")
            if split_handoff_token_n not in (32, 64, 128):
                raise ValueError(
                    "split_handoff_token_n must be 32, 64, or 128."
                )
            if split_handoff_token_n % mma_tiler_mnk[1] != 0:
                raise ValueError(
                    "split handoff tile must be divisible by this role's "
                    f"token N, got {split_handoff_token_n} and "
                    f"{mma_tiler_mnk[1]}."
                )
            if cluster_shape_mnk != (1, 1, 1):
                raise ValueError(
                    "independent split token-N requires cluster (1, 1, 1)."
                )
            if token_padding_block != split_handoff_token_n:
                raise ValueError(
                    "independent split physical token padding must equal the "
                    f"handoff tile, got {token_padding_block} and "
                    f"{split_handoff_token_n}."
                )
            if split_handoff_token_n % split_fc1_token_n != 0:
                raise ValueError(
                    "split handoff tile must be divisible by FC1 token N."
                )
            if split_role == "k1" and mma_tiler_mnk[1] != split_fc1_token_n:
                raise ValueError(
                    "K1 tactic token N must equal split_fc1_token_n."
                )
        if split_role != "fused":
            if (
                not isinstance(split_workspace_counter_tile_tokens, int)
                or isinstance(split_workspace_counter_tile_tokens, bool)
                or split_workspace_counter_tile_tokens <= 0
                or token_padding_block % split_workspace_counter_tile_tokens != 0
            ):
                raise ValueError(
                    "split workspace counter tile must be a positive divisor "
                    f"of token padding, got "
                    f"{split_workspace_counter_tile_tokens!r} and "
                    f"{token_padding_block}."
                )
        if (
            isinstance(split_counter_epoch_banks, bool)
            or not isinstance(split_counter_epoch_banks, int)
            or split_counter_epoch_banks not in (1, 2)
        ):
            raise ValueError(
                "split_counter_epoch_banks must be 1 or 2, got "
                f"{split_counter_epoch_banks!r}."
            )
        if (
            isinstance(split_counter_epoch_bank, bool)
            or not isinstance(split_counter_epoch_bank, int)
            or not 0 <= split_counter_epoch_bank < split_counter_epoch_banks
        ):
            raise ValueError(
                "split_counter_epoch_bank must select an existing bank, got "
                f"bank={split_counter_epoch_bank!r}, "
                f"banks={split_counter_epoch_banks}."
            )
        if split_role == "fused" and (
            split_counter_epoch_banks != 1 or split_counter_epoch_bank != 0
        ):
            raise ValueError(
                "fused MegaMoE cannot use split counter epoch banks."
            )
        self.split_role = split_role
        # Set before ``super`` so the swap-AB base can carry the codegen-time
        # values into its autonomous epilogue construction.
        self.split_fc1_token_n = split_fc1_token_n
        self.split_handoff_token_n = split_handoff_token_n
        self.split_workspace_counter_tile_tokens = (
            split_workspace_counter_tile_tokens
        )
        self.counter_epoch_banks = split_counter_epoch_banks
        self.counter_epoch_bank = split_counter_epoch_bank
        if split_role != "fused":
            role_phase = {"k1": "fc1", "k2": "fc2"}[split_role]
            if execution_phase not in ("fused", role_phase):
                raise ValueError(
                    f"split_role={split_role!r} conflicts with "
                    f"execution_phase={execution_phase!r}."
                )
            execution_phase = role_phase
        if static_expert_shape is None:
            raise NotImplementedError(
                "Sm90MegaMoEFp8Kernel requires "
                "static_expert_shape != None (dynamic-shape MegaMoE is not wired)."
            )
        if hidden != static_expert_shape[2]:
            raise ValueError(
                f"hidden ({hidden}) must equal "
                f"static_expert_shape[2] ({static_expert_shape[2]})."
            )
        if fc2_in_kernel_topk_reduce and not apply_topk_in_fc1:
            raise ValueError(
                "fc2_in_kernel_topk_reduce requires apply_topk_in_fc1=True; "
                "the in-kernel reduce can only atomic-add terms whose topk "
                "score was already absorbed before fc2."
            )

        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
            use_2cta_instrs=use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            sf_padding_block=sf_padding_block,
            load_balance_mode=load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=force_static_sched,
            clc_bundle_size=clc_bundle_size,
            num_sched_stages=num_sched_stages,
            acc_dtype=acc_dtype,
            ab_dtype=ab_dtype,
            sf_vec_size=sf_vec_size,
            fp8_scale_mode=fp8_scale_mode,
            fp8_accum_mode=fp8_accum_mode,
            execution_phase=execution_phase,
            pingpong=pingpong,
            scenario=scenario,
            fc2_in_kernel_topk_reduce=fc2_in_kernel_topk_reduce,
            apply_topk_in_fc1=apply_topk_in_fc1,
            token_back_by_dispatch=token_back_by_dispatch,
            epi_flag_batch=epi_flag_batch,
            gate_up_clamp=gate_up_clamp,
        )

        if split_role != "fused" and not getattr(self, "is_swap_ab", False):
            raise ValueError("split K1/K2 currently require swap-AB.")
        self.phase_mode = {
            "fused": "fc12",
            "k1": "fc1",
            "k2": "fc2",
        }[split_role]
        self.split_publish_fc1_done = split_role == "k1"
        self.split_consume_fc1_done = split_role == "k2"
        self.split_fc1_tile_m = (
            split_fc1_tile_m
            if split_role == "k2"
            else (mma_tiler_mnk[0] if split_role == "k1" else None)
        )
        self.enable_token_comm = True
        self.execution_phase = execution_phase
        self.enable_dispatch_warps = split_role != "k2"
        # SM90 setmaxnreg.*.sync.aligned is warpgroup-synchronous.  Keep one
        # non-compute alignment/register-donor warp after the producer trio so
        # K1/K2 both launch complete 4-warp groups; this is not opposite-phase
        # state and is required even by a purpose-built split role.
        self.enable_empty_warp = True
        dispatch_warp_start = (
            self.empty_warp_id + 1
            if self.enable_empty_warp
            else self.sched_warp_id + 1
        )
        self.dispatch_warp_id = tuple(
            range(dispatch_warp_start, dispatch_warp_start + 4)
        )
        # ``standalone_warps`` dedicates four token-back warps after dispatch;
        # ``reuse_dispatch_warps`` runs the push inline on the dispatch warps.
        self.token_back_standalone = token_back_mode == "standalone_warps"
        token_back_warp_start = dispatch_warp_start + 4
        self.token_back_warp_id = (
            tuple(range(token_back_warp_start, token_back_warp_start + 4))
            if self.token_back_standalone
            else None
        )
        # Keep the established swap-AB and non-swap N=256 budgets. Non-swap
        # N=128 has one epilogue warpgroup and can use the architectural
        # setmaxnreg maximum without approaching the CTA budget.
        self.epi_reg_cnt = 200 if self.token_back_standalone else 216
        if (
            not getattr(self, "is_swap_ab", False)
            and self.wgmma_n_splits == 1
            and not self.pingpong
        ):
            self.epi_reg_cnt = 256
        self.token_back_reg_cnt = 32
        active_dispatch_warp_ids = (
            self.dispatch_warp_id if self.enable_dispatch_warps else ()
        )
        token_back_warp_ids = (
            self.token_back_warp_id if self.token_back_standalone else ()
        )
        empty_warp_ids = (
            (self.empty_warp_id,) if self.enable_empty_warp else ()
        )
        self.threads_per_cta = 32 * len(
            (
                *self.epilogue_warp_id,
                self.tma_a_warp_id,
                self.tma_b_warp_id,
                self.sched_warp_id,
                *empty_warp_ids,
                *active_dispatch_warp_ids,
                *token_back_warp_ids,
            )
        )
        if self.threads_per_cta % 128 != 0:
            raise ValueError(
                "MegaMoE warp roles must form complete SM90 warpgroups; "
                f"got {self.threads_per_cta} threads."
            )
        if (
            self.enable_dispatch_warps
            and self.dispatch_warp_id[0] % 4 != 0
        ):
            raise ValueError(
                "dispatch warps must start on a 4-warp boundary; "
                f"got warp {self.dispatch_warp_id[0]}."
            )
        self.validate_register_policy()

        # Independent MegaMoE-specific constants.
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden
        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.combine_format = CombineFormat.parse("bf16")

        # static_expert_shape = (num_experts_per_rank, intermediate_gateup, hidden).
        self.num_experts_per_rank = static_expert_shape[0]
        self.intermediate_gateup = static_expert_shape[1]
        self.intermediate_downproj = self.intermediate_gateup // 2
        logical_fc2_activation_sf_cols = (
            self.intermediate_downproj // Fp8Fc2ActivationScaleK
        )
        self.fc2_activation_sf_storage_cols = (
            _round_up(logical_fc2_activation_sf_cols, 4)
            if self.fp8_scale_mode in ("blockwise", "mxfp4_hybrid")
            else logical_fc2_activation_sf_cols
        )

        # FP8: 8 bits/elem = 1 byte/element (NVFP4 packs 2 per byte).
        self.hidden_bytes = self.hidden
        # Dispatch pulls scale metadata in uint32 units.  Per-tensor interprets
        # each word as four E8M0 bytes; blockwise interprets it as one FP32
        # scale.  In both modes the atom covers 128 K elements, so dispatch
        # byte plumbing stays unchanged.
        sf_atom_k_elements = Fp8DispatchScaleAtomK
        logical_sf_uint32_per_token = (
            (self.hidden + sf_atom_k_elements - 1) // sf_atom_k_elements
        )
        if self.fp8_scale_mode == "mxfp4_hybrid":
            # One logical FP32 activation scale is replicated across a
            # 16-byte row so dispatch and TMA keep aligned transactions.
            self.sf_uint32_per_token = 4
        else:
            self.sf_uint32_per_token = (
                _round_up(logical_sf_uint32_per_token, 4)
                if self.fp8_scale_mode == "blockwise"
                else logical_sf_uint32_per_token
            )
        # Cross-rank totals: per-rank count * world_size.
        self.num_total_experts = world_size * self.num_experts_per_rank

        is_swap_ab = getattr(self, "is_swap_ab", False)
        self.cluster_tile_tokens = (
            self.mma_tiler_mnk[1] * cluster_shape_mnk[1]
            if is_swap_ab
            else self.mma_tiler_mnk[0] * cluster_shape_mnk[0]
        )

        # Cache region sizing inputs used by workspace layout and __call__.
        (
            self.pool_token_capacity,
            self.pool_sf_capacity,
            self.pool_task_tile_capacity,
        ) = self._pool_shapes()

        # Cohabiting warps before the dispatch group: one or two
        # epilogue/WGMMA warpgroups plus TMA-A/TMA-B/scheduler and the empty
        # old-MMA warp.
        empty_warp_ids = (
            (self.empty_warp_id,) if self.enable_empty_warp else ()
        )
        num_other_warps = len(
            (
                *self.epilogue_warp_id,
                self.tma_a_warp_id,
                self.tma_b_warp_id,
                self.sched_warp_id,
                *empty_warp_ids,
            )
        )

        # For token_back_by_dispatch, the dispatch warp pushes fc2 results
        # from the local pool workspace back to each source rank's internal
        # combine target.
        # Count every FC2 CTA publish in one token cluster tile. Channel tiles
        # are rounded to complete channel clusters, and every CTA along the
        # token-cluster axis independently publishes its output rows.
        if token_back_by_dispatch:
            if is_swap_ab:
                channel_tile = self.mma_tiler_mnk[0]
                channel_cluster = cluster_shape_mnk[0]
                token_cluster = cluster_shape_mnk[1]
            else:
                channel_tile = self.mma_tiler_mnk[1]
                channel_cluster = cluster_shape_mnk[1]
                token_cluster = cluster_shape_mnk[0]
            fc2_publishes = (
                (
                    self.hidden + channel_tile * channel_cluster - 1
                )
                // (channel_tile * channel_cluster)
                * channel_cluster
                * token_cluster
            )
        else:
            fc2_publishes = 0

        self.token_comm = TokenInPullTokenBackPush(
            world_size=self.world_size,
            num_topk=self.num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=self.hidden,
            fc1_token_dtype=self.ab_dtype,
            token_back_by_dispatch=token_back_by_dispatch,
            fc2_publishes_per_token_cluster_tile=fc2_publishes,
            token_back_reduce_topk=(
                token_back_by_dispatch and fc2_in_kernel_topk_reduce
            ),
            token_back_standalone=self.token_back_standalone,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            cluster_tile_tokens=self.cluster_tile_tokens,
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=self.dispatch_warp_id[0],
            num_other_warps=num_other_warps,
            is_swap_ab=is_swap_ab,
            sf_atom_swizzled=(
                self.fp8_scale_mode not in (
                    "blockwise", "mxfp4_hybrid"
                )
            ),
            flag_batch=flag_batch,
            execution_phase=self.execution_phase,
        )

        # Region layout (same call drives both get_workspace_sizes() and the
        # __call__ partition).
        self._local_region_specs = self._build_local_region_specs()
        self._shared_region_specs = self._build_shared_region_specs()
        self._local_offsets, self._local_total = _layout_regions(
            self._local_region_specs
        )
        self._shared_offsets, self._shared_total = _layout_regions(
            self._shared_region_specs
        )
        self._local_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r for r in self._local_region_specs
        }
        self._shared_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r for r in self._shared_region_specs
        }
        local_leading = self._local_offsets["l1_token_buffer"]
        shared_leading = self._shared_offsets["src_token_topk_idx"]
        self.local_zero_i32_count = local_leading // 4
        self.shared_zero_i32_count = shared_leading // 4
        self.local_counter_bank_spans = self._counter_bank_spans(
            self._local_counter_region_specs,
            self._local_offsets,
            data_offset=local_leading,
        )
        self.shared_counter_bank_spans = self._counter_bank_spans(
            self._shared_counter_region_specs,
            self._shared_offsets,
            data_offset=shared_leading,
        )

    def estimated_register_budget(self) -> int:
        """Account only warp roles physically launched by this Mega role."""
        dispatch_warps = (
            len(self.dispatch_warp_id) if self.enable_dispatch_warps else 0
        )
        token_back_warps = (
            len(self.token_back_warp_id)
            if self.token_back_standalone and self.token_back_warp_id
            else 0
        )
        empty_warps = 1 if self.enable_empty_warp else 0
        regs_per_warp = (
            len(self.epilogue_warp_id) * self.epi_reg_cnt
            + self.tma_a_reg_cnt
            + self.tma_b_reg_cnt
            + self.sched_reg_cnt
            + empty_warps * self.empty_reg_cnt
            + dispatch_warps * self.dispatch_reg_cnt
            + token_back_warps * self.token_back_reg_cnt
        )
        return 32 * regs_per_warp

    def sched_ext_fc1_peek_threshold(self) -> int:
        # Peek threshold must match the spin threshold (physical token-N tile)
        # so an early peek hit does not skip the spin and expose stale pool rows.
        return self.cluster_tile_tokens

    # =========================================================================
    # SMEM budget hook (base override)
    # =========================================================================

    def _dispatch_smem_bytes(self) -> int:
        """SMEM for dispatch pull mbarriers, expert scratch, and token buffer.

        Must match ``TokenInPullTokenBackPush.extra_smem_storage_class``:
        ``pull_mbar[Int64, 4] + smem_expert_count[Int32, num_total_experts]
        + pull_buffer[Uint8, 4 * hidden_bytes]``.
        Standalone token-back adds ``tb_pull_mbar[Int64, 4]`` and
        ``tb_pull_buffer[Uint8, 4 * tb_chunk_bytes]``.
        """
        pull_mbar_bytes = _DispatchWarpCount * 8
        expert_count_bytes = self.num_total_experts * 4
        pull_buffer_bytes = _DispatchWarpCount * self.hidden_bytes
        total = (
            _round_up(pull_mbar_bytes, 16)
            + _round_up(expert_count_bytes, 16)
            + _round_up(pull_buffer_bytes, 128)
        )
        if self.token_back_standalone:
            total += (
                _round_up(_DispatchWarpCount * 8, 16)
                + _round_up(
                    _DispatchWarpCount * self.token_comm.tb_chunk_bytes, 128
                )
            )
        return total

    def _smem_misc_budget_bytes(self) -> int:
        """Base misc reservation plus dispatch-warp SMEM."""
        return super()._smem_misc_budget_bytes() + self._dispatch_smem_bytes()

    # =========================================================================
    # Pool sizing (first-principles; identical to the NVFP4 path)
    # =========================================================================

    def _pool_shapes(self) -> Tuple[int, int, int]:
        world_size = self.world_size
        max_tokens_per_rank = self.max_tokens_per_rank
        num_topk = self.num_topk
        num_experts_per_rank = self.num_experts_per_rank
        token_padding_block = self.token_padding_block
        sf_padding_block = self.sf_padding_block
        counter_tile_tokens = (
            self.split_workspace_counter_tile_tokens
            if self.split_workspace_counter_tile_tokens is not None
            else self.cluster_tile_tokens
        )

        max_recv = world_size * max_tokens_per_rank
        max_per_token = min(num_topk, num_experts_per_rank)
        raw = (
            max_recv * max_per_token
            + num_experts_per_rank * (token_padding_block - 1)
        )
        pool_token_capacity = _round_up(raw, token_padding_block)
        pool_sf_capacity = (
            (pool_token_capacity // token_padding_block) * sf_padding_block
        )
        pool_task_tile_capacity = (
            (pool_token_capacity + counter_tile_tokens - 1)
            // counter_tile_tokens
            + num_experts_per_rank
        )
        return (
            pool_token_capacity,
            pool_sf_capacity,
            pool_task_tile_capacity,
        )

    # =========================================================================
    # Region tables
    # =========================================================================

    def _counter_region_name(
        self, logical_name: str, bank: Optional[int] = None
    ) -> str:
        selected = self.counter_epoch_bank if bank is None else bank
        return (
            logical_name
            if selected == 0
            else f"{logical_name}__bank{selected}"
        )

    def _expand_counter_region_specs(
        self, specs: List[_RegionSpec]
    ) -> List[_RegionSpec]:
        if self.counter_epoch_banks == 1:
            return specs
        return [
            _RegionSpec(
                self._counter_region_name(spec.name, bank),
                spec.cute_dtype,
                spec.shape,
                spec.align,
            )
            for bank in range(self.counter_epoch_banks)
            for spec in specs
        ]

    def _counter_bank_spans(
        self,
        counter_specs: Tuple[_RegionSpec, ...],
        offsets: Dict[str, int],
        *,
        data_offset: int,
    ) -> Tuple[Tuple[int, int], ...]:
        if self.counter_epoch_banks == 1:
            return ((0, data_offset),)
        relative_offsets, bank_span = _layout_regions(list(counter_specs))
        spans = []
        for bank in range(self.counter_epoch_banks):
            bank_offset = bank * bank_span
            for spec in counter_specs:
                physical_name = self._counter_region_name(spec.name, bank)
                expected = bank_offset + relative_offsets[spec.name]
                actual = offsets[physical_name]
                if actual != expected:
                    raise RuntimeError(
                        f"counter bank layout mismatch for {physical_name}: "
                        f"expected {expected}, got {actual}"
                    )
            spans.append((bank_offset, bank_span))
        return tuple(spans)

    def _build_local_region_specs(self) -> List[_RegionSpec]:
        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        pool_task_tile_capacity = self.pool_task_tile_capacity
        num_experts_per_rank = self.num_experts_per_rank
        num_total_experts = self.num_total_experts
        hidden_bytes = self.hidden_bytes
        sf_uint32_per_token = self.sf_uint32_per_token
        intermediate_downproj = self.intermediate_downproj
        sf_padding_block = self.sf_padding_block

        sf_total_rows_upper = (
            pool_token_capacity + num_experts_per_rank * sf_padding_block
        )
        per_tensor_sf_cols = intermediate_downproj // Fp8E8M0SfVecSize
        sf_block_cols = (
            ((per_tensor_sf_cols + 3) // 4) * 4
        )
        cluster_token_ctas = (
            self.cluster_shape_mn[1]
            if getattr(self, "is_swap_ab", False)
            else self.cluster_shape_mn[0]
        )
        fc1_done_slots = pool_task_tile_capacity * cluster_token_ctas

        # Accumulating counters are front-placed so kernel_tail can reset them
        # as one contiguous Int32 prefix before the next launch.
        counter_specs: List[_RegionSpec] = [
            _RegionSpec(
                "l1_arrival_count",
                cutlass.Int32,
                (pool_task_tile_capacity,),
                16,
            ),
            _RegionSpec(
                "expert_send_count",
                cutlass.Int64,
                (num_total_experts,),
                16,
            ),
            _RegionSpec(
                "grid_sync_counter",
                cutlass.Int32,
                (_GridSyncSlotCount,),
                16,
            ),
            _RegionSpec(
                "fc1_done_counter",
                cutlass.Int32,
                (fc1_done_slots,),
                16,
            ),
        ]

        if self.split_role != "fused":
            counter_specs.append(
                _RegionSpec(
                    "split_dispatch_ready",
                    cutlass.Int32,
                    (1,),
                    16,
                )
            )
        if self.token_back_by_dispatch:
            counter_specs.append(
                _RegionSpec(
                    "fc2_done_counter",
                    cutlass.Int32,
                    (num_experts_per_rank,),
                    16,
                )
            )

        if self.load_balance_mode == "atomic_counter":
            counter_specs.append(
                _RegionSpec(
                    "load_balance_counter",
                    cutlass.Int32,
                    (2,),
                    16,
                )
            )

        self._local_counter_region_specs = tuple(counter_specs)
        specs = self._expand_counter_region_specs(counter_specs)
        # Data buffers start at l1_token_buffer. The persistent NVLink phase
        # counter is intentionally after the reset prefix.
        specs += [
            _RegionSpec(
                "l1_token_buffer",
                cutlass.Uint8,
                (pool_token_capacity, hidden_bytes),
                128,
            ),
            _RegionSpec(
                "nvlink_barrier_counter",
                cutlass.Int32,
                (1,),
                16,
            ),
            _RegionSpec(
                "l1_sf_buffer",
                cutlass.Int32,
                (pool_sf_capacity * sf_uint32_per_token,),
                16,
            ),
            _RegionSpec(
                "l1_topk_weights_buffer",
                cutlass.Float32,
                (pool_token_capacity,),
                16,
            ),
            _RegionSpec(
                "token_src_metadata",
                cutlass.Uint8,
                (pool_token_capacity, _TokenMetadataBytes),
                16,
            ),
            _RegionSpec(
                "fc1_output",
                self.ab_dtype,
                (pool_token_capacity, intermediate_downproj),
                128,
            ),
            _RegionSpec(
                "fc1_output_sf",
                cutlass.Float32
                if self.fp8_scale_mode in (
                    "blockwise", "mxfp4_hybrid"
                )
                else cutlass.Float8E8M0FNU,
                (
                    (pool_token_capacity, self.fc2_activation_sf_storage_cols)
                    if self.fp8_scale_mode in (
                        "blockwise", "mxfp4_hybrid"
                    )
                    else (sf_total_rows_upper, sf_block_cols)
                ),
                128,
            ),
        ]

        if self.token_back_by_dispatch:
            specs.append(
                _RegionSpec(
                    "fc2_output_workspace",
                    cutlass.BFloat16,
                    (pool_token_capacity, 1, self.hidden),
                    128,
                )
            )

        return specs

    def _build_shared_region_specs(self) -> List[_RegionSpec]:
        world_size = self.world_size
        num_topk = self.num_topk
        max_tokens_per_rank = self.max_tokens_per_rank
        num_experts_per_rank = self.num_experts_per_rank

        max_slot = max_tokens_per_rank * num_topk

        counter_specs = [
            _RegionSpec(
                "expert_recv_count",
                cutlass.Int64,
                (world_size, num_experts_per_rank),
                16,
            ),
            _RegionSpec(
                "expert_recv_count_sum",
                cutlass.Int64,
                (num_experts_per_rank,),
                16,
            ),
        ]
        if self.split_role != "fused":
            # K3's cross-rank join counter must be peer-addressable and part
            # of the shared reset prefix. Each rank publishes local K2
            # completion to every peer before any rank reduces its remotely
            # populated combine plane.
            counter_specs.append(
                _RegionSpec("split_k2_join_count", cutlass.Int32, (1,), 16)
            )
        self._shared_counter_region_specs = tuple(counter_specs)
        specs = self._expand_counter_region_specs(counter_specs)
        specs += [
            _RegionSpec(
                "src_token_topk_idx",
                cutlass.Int32,
                (num_experts_per_rank, world_size, max_slot),
                16,
            ),
            _RegionSpec(
                "nvlink_barrier_signal",
                cutlass.Int32,
                (_NvlinkSlotCount,),
                16,
            ),
        ]

        # The per-topk FC2 plane is an implementation workspace, not public IO.
        # It is the cross-rank STG/TMA target and therefore belongs in the shared
        # symmetric workspace. In-kernel reduce accumulates directly into
        # output_activation (REDG from epi_warps, or bulk reduce push from the
        # dispatch token-back modes) and needs no staging.
        if not self.fc2_in_kernel_topk_reduce:
            specs.append(
                _RegionSpec(
                    "combine_quant",
                    self.combine_format.act_dtype,
                    (max_tokens_per_rank, num_topk, self.hidden),
                    128,
                )
            )
        return specs

    # =========================================================================
    # Public: workspace size query
    # =========================================================================

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return ``(local_ws_bytes, shared_ws_bytes)``."""
        return self._local_total, self._shared_total

    # =========================================================================
    # Workspace partition helpers (mirror the NVFP4 mega kernel)
    # =========================================================================

    @staticmethod
    def _make_typed_view(
        byte_workspace: cute.Pointer,
        byte_offset: int,
        cute_dtype: Any,
        shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]],
        assumed_align: int,
    ) -> cute.Tensor:
        """Build a typed view at a 64-bit byte offset from an opaque base."""
        byte_ptr = byte_workspace + Int64(byte_offset)
        typed_iter = cute.make_ptr(
            cute_dtype,
            byte_ptr.toint(),
            AddressSpace.gmem,
            assumed_align=assumed_align,
        )
        return cute.make_tensor(typed_iter, cute.make_layout(shape, stride=stride))

    def _view_local(
        self,
        local_workspace: cute.Pointer,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        return self._partition_region(
            local_workspace,
            self._local_offsets,
            self._local_region_by_name[name],
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    def _view_shared(
        self,
        shared_workspace: cute.Pointer,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        return self._partition_region(
            shared_workspace,
            self._shared_offsets,
            self._shared_region_by_name[name],
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    def _view_local_split_counter(
        self,
        local_workspace: cute.Pointer,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        return self._view_local(
            local_workspace,
            self._counter_region_name(name),
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    def _view_shared_split_counter(
        self,
        shared_workspace: cute.Pointer,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        return self._view_shared(
            shared_workspace,
            self._counter_region_name(name),
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )
    def _partition_region(
        self,
        byte_workspace: cute.Pointer,
        offsets: Dict[str, int],
        spec: _RegionSpec,
        *,
        cute_dtype: Optional[Any],
        shape: Optional[Tuple[int, ...]],
        stride: Optional[Tuple[int, ...]],
    ) -> cute.Tensor:
        dt = cute_dtype if cute_dtype is not None else spec.cute_dtype
        sh = shape if shape is not None else spec.shape
        st = stride
        if st is None:
            if cute_dtype is None and shape is None:
                st = spec.stride_row_major
            else:
                out: List[int] = [1]
                for d in reversed(list(sh)[1:]):
                    out.append(out[-1] * d)
                out.reverse()
                st = tuple(out)
        return self._make_typed_view(
            byte_workspace, offsets[spec.name], dt, sh, st, spec.align,
        )

    # =========================================================================
    # Split role entrypoints
    # =========================================================================

    @cute.jit
    def split_k1_entry(
        self,
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc1_weight_dequant_scale: cute.Tensor,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper_host,
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Strict dispatch + FC1 ABI with no FC2 model state."""
        if cutlass.const_expr(self.split_role != "k1"):
            raise ValueError("split_k1_entry requires split_role='k1'.")
        if cutlass.const_expr(self.fp8_scale_mode != "mxfp4_hybrid"):
            raise ValueError("split K1 requires mxfp4_hybrid scaling.")
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        if cutlass.const_expr(self.world_size == 1):
            peer_rank_ptr_mapper = SingleRankSymBufferDevice()
            local_rank = Int32(self.local_rank)
        else:
            if cutlass.const_expr(peer_rank_ptr_mapper_host is None):
                raise ValueError("multi-rank split K1 requires a peer mapper")
            peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()
            local_rank = peer_rank_ptr_mapper_host.rank_idx

        l1_token_buffer_u8 = self._view_local(
            local_workspace, "l1_token_buffer",
        )
        l1_token_buffer_fp8 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            self.ab_dtype,
            (self.pool_token_capacity, self.hidden),
            (self.hidden, 1),
            self._local_region_by_name["l1_token_buffer"].align,
        )
        l1_sf_buffer_i32 = self._view_local(local_workspace, "l1_sf_buffer")
        l1_sf_buffer_fp32 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_sf_buffer"],
            cutlass.Float32,
            (self.pool_sf_capacity, self.sf_uint32_per_token),
            (self.sf_uint32_per_token, 1),
            self._local_region_by_name["l1_sf_buffer"].align,
        )
        l1_topk_weights = self._view_local(
            local_workspace, "l1_topk_weights_buffer",
        )
        l1_arrival_count = self._view_local_split_counter(
            local_workspace, "l1_arrival_count",
        )
        token_src_metadata = self._view_local(
            local_workspace, "token_src_metadata",
        )
        expert_send_count = self._view_local_split_counter(
            local_workspace, "expert_send_count",
        )
        grid_sync_counter = self._view_local_split_counter(
            local_workspace, "grid_sync_counter",
        )
        nvlink_barrier_counter = self._view_local(
            local_workspace, "nvlink_barrier_counter",
        )
        fc1_output = self._view_local(local_workspace, "fc1_output")
        fc1_output_sf = self._view_local(local_workspace, "fc1_output_sf")
        fc1_done_counter = self._view_local_split_counter(
            local_workspace, "fc1_done_counter",
        )
        split_dispatch_ready = self._view_local_split_counter(
            local_workspace, "split_dispatch_ready",
        )
        load_balance_counter = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            load_balance_counter = self._view_local_split_counter(
                local_workspace, "load_balance_counter",
            )

        src_token_topk_idx = self._view_shared(
            shared_workspace, "src_token_topk_idx",
        )
        expert_recv_count = self._view_shared_split_counter(
            shared_workspace, "expert_recv_count",
        )
        expert_recv_count_sum = self._view_shared_split_counter(
            shared_workspace, "expert_recv_count_sum",
        )
        nvlink_barrier_signal = self._view_shared(
            shared_workspace, "nvlink_barrier_signal",
        )
        expert_token_sizes = self._view_shared_split_counter(
            shared_workspace,
            "expert_recv_count_sum",
            cute_dtype=cutlass.Int32,
            shape=(self.num_experts_per_rank,),
            stride=(2,),
        )
        local_bank_offset, local_bank_bytes = self.local_counter_bank_spans[
            self.counter_epoch_bank
        ]
        shared_bank_offset, shared_bank_bytes = self.shared_counter_bank_spans[
            self.counter_epoch_bank
        ]
        local_zero_prefix = self._make_typed_view(
            local_workspace,
            local_bank_offset,
            cutlass.Int32,
            (local_bank_bytes // 4,),
            (1,),
            16,
        )
        shared_zero_prefix = self._make_typed_view(
            shared_workspace,
            shared_bank_offset,
            cutlass.Int32,
            (shared_bank_bytes // 4,),
            (1,),
            16,
        )
        token_comm_args = SplitK1TokenCommArgs(
            input_token_buffer=activation,
            input_sf_buffer=activation_sf,
            topk_idx=topk_idx,
            input_topk_weights_buffer=topk_weights,
            expert_send_count=expert_send_count,
            expert_recv_count=expert_recv_count,
            expert_recv_count_sum=expert_recv_count_sum,
            src_token_topk_idx=src_token_topk_idx,
            fc1_input_token_buffer=l1_token_buffer_u8,
            fc1_input_sf_buffer=l1_sf_buffer_i32,
            fc1_input_topk_weights_buffer=l1_topk_weights,
            fc1_ready_counter=l1_arrival_count,
            token_src_metadata=token_src_metadata,
            nvlink_barrier_signal=nvlink_barrier_signal,
            nvlink_barrier_counter=nvlink_barrier_counter,
            grid_sync_counter=grid_sync_counter,
            local_zero_prefix=local_zero_prefix,
            shared_zero_prefix=shared_zero_prefix,
            split_dispatch_ready=split_dispatch_ready,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            world_size=self.world_size,
            local_rank=local_rank,
            num_total_experts=self.num_total_experts,
            num_experts_per_rank=self.num_experts_per_rank,
            num_topk=self.num_topk,
            hidden_bytes=self.hidden_bytes,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            sm_count=sm_count,
        )
        Sm90SwapABSwigluFp8Fc12Kernel.launch_fc1_only(
            self,
            activation=l1_token_buffer_fp8,
            fc1_weight=fc1_weight,
            activation_sf=l1_sf_buffer_fp32,
            fc1_weight_sf=fc1_weight_sf,
            fc1_weight_dequant_scale=fc1_weight_dequant_scale,
            fc1_output=fc1_output,
            fc1_output_sf=fc1_output_sf,
            topk_scores=l1_topk_weights,
            fc1_done_counter=fc1_done_counter,
            offs=None,
            max_active_clusters=max_active_clusters,
            stream=stream,
            load_balance_counter=load_balance_counter,
            expert_token_sizes=expert_token_sizes,
            token_comm_args=token_comm_args,
        )

    @cute.jit
    def split_k2_entry(
        self,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc2_weight_dequant_scale: cute.Tensor,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper_host,
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Strict FC2 + direct-combine ABI with no FC1 model state."""
        if cutlass.const_expr(self.split_role != "k2"):
            raise ValueError("split_k2_entry requires split_role='k2'.")
        if cutlass.const_expr(self.fp8_scale_mode != "mxfp4_hybrid"):
            raise ValueError("split K2 requires mxfp4_hybrid scaling.")

        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        if cutlass.const_expr(self.world_size == 1):
            peer_rank_ptr_mapper = SingleRankSymBufferDevice()
            local_rank = Int32(self.local_rank)
        else:
            if cutlass.const_expr(peer_rank_ptr_mapper_host is None):
                raise ValueError("multi-rank split K2 requires a peer mapper")
            peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()
            local_rank = peer_rank_ptr_mapper_host.rank_idx

        fc1_output = self._view_local(local_workspace, "fc1_output")
        fc1_output_sf = self._view_local(local_workspace, "fc1_output_sf")
        fc1_done_counter = self._view_local_split_counter(
            local_workspace, "fc1_done_counter",
        )
        split_dispatch_ready = self._view_local_split_counter(
            local_workspace, "split_dispatch_ready",
        )
        token_src_metadata = self._view_local(
            local_workspace, "token_src_metadata",
        )
        combine_output = self._view_shared(
            shared_workspace, "combine_quant",
        )
        expert_token_sizes = self._view_shared_split_counter(
            shared_workspace,
            "expert_recv_count_sum",
            cute_dtype=cutlass.Int32,
            shape=(self.num_experts_per_rank,),
            stride=(2,),
        )
        load_balance_counter = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            load_balance_counter = self._view_local_split_counter(
                local_workspace, "load_balance_counter",
            )

        token_comm_args = SplitK2TokenCommArgs(
            token_src_metadata=token_src_metadata,
            combine_output=combine_output,
            split_dispatch_ready=split_dispatch_ready,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
        )
        Sm90SwapABSwigluFp8Fc12Kernel.launch_fc2_only(
            self,
            fc1_output=fc1_output,
            fc1_output_sf=fc1_output_sf,
            fc1_done_counter=fc1_done_counter,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            fc2_weight_dequant_scale=fc2_weight_dequant_scale,
            fc2_output=combine_output,
            offs=None,
            max_active_clusters=max_active_clusters,
            stream=stream,
            load_balance_counter=load_balance_counter,
            expert_token_sizes=expert_token_sizes,
            token_comm_args=token_comm_args,
        )

    # =========================================================================
    # Legacy fused entrypoint
    # =========================================================================

    @cute.jit
    def __call__(
        self,
        # Scale ABI notation: T=tokens, E=local experts, H=hidden, and I=the
        # down-projection width. FC1 produces the gate/up width 2I.
        # User-domain inputs (peer-mapped on the symmetric heap).
        activation: cute.Tensor,           # (T, hidden) fp8
        # per_tensor: (T, round_up(ceil(H/32), 4)) E8M0 metadata; dispatched,
        # but not used by GEMM dequantization.
        # blockwise: storage (T, round_up(H/128, 4)); the first H/128 FP32
        # activation scales are used by FC1.
        activation_sf: cute.Tensor,
        topk_idx: cute.Tensor,             # (T, num_topk) Int64
        topk_weights: cute.Tensor,         # (T, num_topk) Float32
        # Per-rank model weights (local-only; not in workspace).
        fc1_weight: cute.Tensor,            # (E, H, 2I) FP8; both modes
        # per_tensor: (E, flat_sf) padded/swizzled E8M0 placeholder, unused.
        # blockwise: (E, 2I/128, H/128) FP32 weight scales, used by FC1.
        fc1_weight_sf: cute.Tensor,
        # per_tensor: (1,) FP32, used by FC1; blockwise: (1,) ones, unused.
        fc1_activation_dequant_scale: cute.Tensor,
        # per_tensor: (E,) FP32, used by FC1; blockwise: (E,) ones, unused.
        fc1_weight_dequant_scale: cute.Tensor,
        fc2_weight: cute.Tensor,            # (E, I, H) FP8; both modes
        # per_tensor: (E, flat_sf) padded/swizzled E8M0 placeholder, unused.
        # blockwise: (E, H/128, I/128) FP32 weight scales, used by FC2.
        fc2_weight_sf: cute.Tensor,
        # per_tensor: (1,) FP32, used to quantize FC2 input and dequantize FC2;
        # blockwise: (1,) ones, unused; FC2 uses internal per-token block scales.
        fc2_activation_dequant_scale: cute.Tensor,
        # per_tensor: (E,) FP32, used by FC2; blockwise: (E,) ones, unused.
        fc2_weight_dequant_scale: cute.Tensor,
        # Final combined output consumed by the caller.
        output_activation: cute.Tensor,    # (T, hidden) BF16
        # Opaque workspaces.
        local_workspace: cute.Pointer,     # uint8 gmem base of local_ws_bytes
        shared_workspace: cute.Pointer,    # uint8 gmem base of shared_ws_bytes
        # Runtime host payload; packed into ``SymBuffer{world_size}``.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Launch the FP8 MegaMoE-complete fused kernel.

        Pointer-mapping contract mirrors the NVFP4 path:
          * ``activation`` / ``activation_sf`` / ``topk_weights`` MUST point
            into memory reachable via
            ``peer_rank_ptr_mapper.ptr_map_to_rank(...)`` (NVSHMEM symmetric
            heap).  Single-rank degenerate runs are allowed.
          * ``topk_idx`` is read on the local rank only.
          * ``fc1_weight`` / ``fc1_weight_sf`` / ``fc2_weight`` /
            ``fc2_weight_sf`` are local-only.

          * Under in-kernel reduce (``fc2_in_kernel_topk_reduce``),
            ``output_activation`` is the cross-rank accumulate target (REDG or
            bulk reduce push) and must also be peer reachable. Under separate
            reduce, peer writes target the internal ``combine_quant``
            shared-workspace region and ``output_activation`` may be
            rank-local memory.
        """
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()

        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        hidden = self.hidden
        scale_metadata_bytes_per_token = self.sf_uint32_per_token * 4

        # L1 token buffer: Uint8 view (dispatch_pull byte arith) + fp8 view
        # (fc1 GEMM mainloop).  Same byte offset.
        l1_token_buffer_u8 = self._view_local(local_workspace, "l1_token_buffer")
        l1_token_buffer_fp8 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            self.ab_dtype,
            (pool_token_capacity, hidden),
            (hidden, 1),
            self._local_region_by_name["l1_token_buffer"].align,
        )

        # L1 SF buffer: Int32 view is dispatch_pull's wire format.  Per-tensor
        # keeps the atom-swizzled E8M0 view; blockwise stores one FP32 scale per
        # uint32 word in row-major order.
        l1_sf_buffer_i32 = self._view_local(local_workspace, "l1_sf_buffer")
        if cutlass.const_expr(
            self.fp8_scale_mode in ("blockwise", "mxfp4_hybrid")
        ):
            l1_sf_buffer_for_fc1 = self._make_typed_view(
                local_workspace,
                self._local_offsets["l1_sf_buffer"],
                cutlass.Float32,
                (pool_sf_capacity, self.sf_uint32_per_token),
                (self.sf_uint32_per_token, 1),
                self._local_region_by_name["l1_sf_buffer"].align,
            )
        else:
            l1_sf_buffer_for_fc1 = self._make_typed_view(
                local_workspace,
                self._local_offsets["l1_sf_buffer"],
                cutlass.Float8E8M0FNU,
                (pool_sf_capacity, scale_metadata_bytes_per_token),
                (scale_metadata_bytes_per_token, 1),
                self._local_region_by_name["l1_sf_buffer"].align,
            )

        l1_topk_weights_buffer = self._view_local(
            local_workspace, "l1_topk_weights_buffer",
        )
        l1_arrival_count = self._view_local(local_workspace, "l1_arrival_count")
        # token_src_metadata storage = (pool_token_capacity, TokenSrcMetadata.nbytes) Uint8;
        # dispatch_pull writes one packed Int64 per pool token row (see TokenSrcMetadata).
        token_src_metadata = self._view_local(
            local_workspace, "token_src_metadata",
        )
        expert_send_count = self._view_local(local_workspace, "expert_send_count")
        grid_sync_counter = self._view_local(local_workspace, "grid_sync_counter")
        split_dispatch_ready = None
        if cutlass.const_expr(self.split_role != "fused"):
            split_dispatch_ready = self._view_local(
                local_workspace, "split_dispatch_ready",
            )
        nvlink_barrier_counter = self._view_local(
            local_workspace, "nvlink_barrier_counter",
        )
        fc1_output = self._view_local(local_workspace, "fc1_output")
        fc1_output_sf = self._view_local(local_workspace, "fc1_output_sf")
        fc1_done_counter = self._view_local(local_workspace, "fc1_done_counter")

        load_balance_counter: Optional[cute.Tensor] = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            load_balance_counter = self._view_local(
                local_workspace, "load_balance_counter",
            )
            phase_counter_idx = 1 if self.execution_phase == "fc2" else 0
            load_balance_counter = cute.make_tensor(
                load_balance_counter.iterator + phase_counter_idx,
                cute.make_layout(1),
            )

        # MoE-domain cross-rank combine target. Separate reduce stages one
        # result per (token, topk) in workspace; in-kernel reduce aliases the
        # public 2D output because writers collapse topk on the fly (epi_warps
        # REDG, or the dispatch modes' cp.reduce bulk push).
        if cutlass.const_expr(self.fc2_in_kernel_topk_reduce):
            combine_target = cute.make_tensor(
                output_activation.iterator,
                cute.make_layout(
                    (self.max_tokens_per_rank, 1, hidden),
                    stride=(hidden, hidden, 1),
                ),
            )
        else:
            combine_target = self._view_shared(shared_workspace, "combine_quant")

        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_workspace_native = self._view_local(
                local_workspace, "fc2_output_workspace",
            )
            fc2_output_workspace_u8 = self._make_typed_view(
                local_workspace,
                self._local_offsets["fc2_output_workspace"],
                cutlass.Uint8,
                (pool_token_capacity * hidden * 2,),
                None,
                self._local_region_by_name["fc2_output_workspace"].align,
            )
            fc2_done_counter = self._view_local(local_workspace, "fc2_done_counter")
            combine_output_comm = cute.recast_tensor(
                combine_target, cutlass.Uint8,
            )
            fc2_output_target = fc2_output_workspace_native
        else:
            fc2_output_workspace_native = None
            fc2_output_workspace_u8 = None
            fc2_done_counter = None
            combine_output_comm = combine_target
            fc2_output_target = combine_target

        # Shared regions.
        src_token_topk_idx = self._view_shared(
            shared_workspace, "src_token_topk_idx",
        )
        expert_recv_count = self._view_shared(shared_workspace, "expert_recv_count")
        expert_recv_count_sum = self._view_shared(
            shared_workspace, "expert_recv_count_sum",
        )
        nvlink_barrier_signal = self._view_shared(
            shared_workspace, "nvlink_barrier_signal",
        )

        # i32 stride=(2,) view onto the i64 ``expert_recv_count_sum`` buffer --
        # low32 bits hold per-expert total token count after _dispatch_barrier;
        # zero-copy alias for sizes-mode scheduling.
        expert_token_sizes = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
            cute_dtype=cutlass.Int32,
            shape=(self.num_experts_per_rank,),
            stride=(2,),
        )
        local_zero_prefix = self._make_typed_view(
            local_workspace,
            0,
            cutlass.Int32,
            (self.local_zero_i32_count,),
            (1,),
            16,
        )
        shared_zero_prefix = self._make_typed_view(
            shared_workspace,
            0,
            cutlass.Int32,
            (self.shared_zero_i32_count,),
            (1,),
            16,
        )

        token_comm_args = ExtractedTokenCommArgs(
            input_token_buffer=activation,
            input_sf_buffer=activation_sf,
            topk_idx=topk_idx,
            input_topk_weights_buffer=topk_weights,
            expert_send_count=expert_send_count,
            expert_recv_count=expert_recv_count,
            expert_recv_count_sum=expert_recv_count_sum,
            src_token_topk_idx=src_token_topk_idx,
            fc1_input_token_buffer=l1_token_buffer_u8,
            fc1_input_sf_buffer=l1_sf_buffer_i32,
            fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
            fc1_ready_counter=l1_arrival_count,
            token_src_metadata=token_src_metadata,
            combine_output=combine_output_comm,
            fc2_output_workspace=fc2_output_workspace_u8,
            fc2_done_counter=fc2_done_counter,
            nvlink_barrier_signal=nvlink_barrier_signal,
            nvlink_barrier_counter=nvlink_barrier_counter,
            grid_sync_counter=grid_sync_counter,
            split_dispatch_ready=split_dispatch_ready,
            local_zero_prefix=local_zero_prefix,
            shared_zero_prefix=shared_zero_prefix,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            world_size=self.world_size,
            local_rank=peer_rank_ptr_mapper_host.rank_idx,
            num_total_experts=self.num_total_experts,
            num_experts_per_rank=self.num_experts_per_rank,
            num_topk=self.num_topk,
            hidden_bytes=self.hidden_bytes,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            sm_count=sm_count,
        )

        _fc12_kwargs = dict(
            activation=l1_token_buffer_fp8,
            fc1_weight=fc1_weight,
            activation_sf=l1_sf_buffer_for_fc1,
            fc1_weight_sf=fc1_weight_sf,
            fc1_activation_dequant_scale=fc1_activation_dequant_scale,
            fc1_weight_dequant_scale=fc1_weight_dequant_scale,
            fc1_output=fc1_output,
            fc1_output_sf=fc1_output_sf,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            fc2_activation_dequant_scale=fc2_activation_dequant_scale,
            fc2_weight_dequant_scale=fc2_weight_dequant_scale,
            fc2_output=fc2_output_target,
            topk_scores=l1_topk_weights_buffer,
            fc1_done_counter=fc1_done_counter,
            offs=None,
            max_active_clusters=max_active_clusters,
            stream=stream,
            load_balance_counter=load_balance_counter,
            expert_token_sizes=expert_token_sizes,
            token_comm_args=token_comm_args,
        )
        if cutlass.const_expr(getattr(self, "is_swap_ab", False)):
            Sm90SwapABSwigluFp8Fc12Kernel.__call__(self, **_fc12_kwargs)
        else:
            Sm90SwigluFp8Fc12Kernel.__call__(self, **_fc12_kwargs)

        # Match the NVFP4/MXFP8 compute graphs: deepgemm folds routing weights
        # into the SwiGLU output before FC1-output quantization, while the
        # transformers graph leaves each term unweighted and applies scores in
        # this standalone reducer.
        if cutlass.const_expr(
            not self.fc2_in_kernel_topk_reduce
            and self.split_role == "fused"
        ):
            score = (
                topk_weights if cutlass.const_expr(not self.apply_topk_in_fc1)
                else None
            )
            TopkReduce(
                self.hidden,
                self.num_topk,
                self.combine_format,
                sm_arch=get_cutedsl_target_arch(),
            )(
                combine_target,
                None,
                output_activation,
                topk_idx,
                score,
                stream,
            )

    # =========================================================================
    # TokenComm delegation surface consumed by the fc1/fc2 base kernel
    # =========================================================================

    def token_comm_extra_smem_storage_class(self) -> type:
        if self.split_role == "k2":
            return None
        return self.token_comm.extra_smem_storage_class()

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        if self.split_role == "k2":
            return None
        return self.token_comm.fc1_ready_counter_ptr(token_comm_args)

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        if cutlass.const_expr(self.split_role == "k2"):
            ready_ptr = token_comm_args.split_dispatch_ready.iterator
            spin_wait(
                ready_ptr,
                lambda value: value >= Int32(1),
                fail_sleep_cycles=20,
            )
            cute.arch.load(
                ready_ptr, Int32, sem="acquire", scope="sys",
            )
            cute.arch.fence_acq_rel_sys()
        else:
            self.token_comm.sched_warp_pre_init_wait(token_comm_args)

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self, token_comm_args, work_tile_info,
    ):
        self.token_comm.fc1_tma_b_predispatch_spin(
            token_comm_args, work_tile_info,
        )

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        if cutlass.const_expr(self.execution_phase == "fc2"):
            self.token_comm.split_fc2_dispatch_warp_body(
                token_comm_args,
                token_comm_storage,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )
        else:
            self.token_comm.dispatch_warp_body(
                token_comm_args,
                token_comm_storage,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )

    @cute.jit
    def token_comm_hook_token_back_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        self.token_comm.token_back_warp_body(
            token_comm_args,
            token_comm_storage,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )

    @cute.jit
    def token_comm_hook_tail_reset_shared_counters(
        self,
        token_comm_args,
        *,
        cta_linear_id,
        local_warp_idx,
        lane_idx,
    ):
        self.token_comm.tail_reset_shared_counters(
            token_comm_args,
            cta_linear_id=cta_linear_id,
            local_warp_idx=local_warp_idx,
            lane_idx=lane_idx,
        )

    @cute.jit
    def token_comm_hook_kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        if cutlass.const_expr(self.split_role == "fused"):
            self.token_comm.kernel_tail(
                token_comm_args,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )


class Sm90MegaMoESwapABFp8Kernel(
    Sm90MegaMoEFp8Kernel,
    Sm90SwapABSwigluFp8Fc12Kernel,
):
    """MegaMoE wiring that reuses token communication with the swap-AB base."""

    pass


class Sm90MegaMoESwapABMxfp4Fp8Kernel(
    Sm90MegaMoEFp8Kernel,
    Sm90SwapABSwigluMxfp4Fp8Fc12Kernel,
):
    """MegaMoE with the packed MXFP4 RS FC12 specialization."""

    pass


class MegaMoEFp8TopkReduceLauncher:
    """Launch standalone K3 from the shared workspace after Green Contexts join."""

    def __init__(self, mega_kernel: Sm90MegaMoEFp8Kernel) -> None:
        if mega_kernel.fc2_in_kernel_topk_reduce:
            raise ValueError("standalone TopK reduce is disabled for in-kernel reduce")
        combine_spec = mega_kernel._shared_region_by_name["combine_quant"]
        self.combine_offset = mega_kernel._shared_offsets["combine_quant"]
        self.combine_align = combine_spec.align
        self.max_tokens_per_rank = mega_kernel.max_tokens_per_rank
        self.num_topk = mega_kernel.num_topk
        self.hidden = mega_kernel.hidden
        self.combine_format = mega_kernel.combine_format
        self.apply_topk_in_fc1 = mega_kernel.apply_topk_in_fc1

    @cute.jit
    def __call__(
        self,
        shared_workspace: cute.Pointer,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        output_activation: cute.Tensor,
        stream,
    ) -> None:
        combine_target = Sm90MegaMoEFp8Kernel._make_typed_view(
            shared_workspace,
            self.combine_offset,
            self.combine_format.act_dtype,
            (self.max_tokens_per_rank, self.num_topk, self.hidden),
            (self.num_topk * self.hidden, self.hidden, 1),
            self.combine_align,
        )
        score = None if self.apply_topk_in_fc1 else topk_weights
        TopkReduce(
            self.hidden,
            self.num_topk,
            self.combine_format,
            sm_arch=get_cutedsl_target_arch(),
        )(
            combine_target,
            None,
            output_activation,
            topk_idx,
            score,
            stream,
        )
