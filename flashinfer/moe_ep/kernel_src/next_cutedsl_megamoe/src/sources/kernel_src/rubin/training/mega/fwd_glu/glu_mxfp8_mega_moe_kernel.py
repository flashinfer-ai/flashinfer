# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Full MegaMoE (multi-rank) mxfp8 GLU training-forward kernel."""

from typing import Any, Literal, Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from ......api import (
    ImplDesc,
    KernelClass,
    OptionalRequirement,
    ProblemDesc,
    StaticOrRuntimeIntegerType,
)
from ......helpers.device_workspace import DeviceWorkspace
from ......helpers.smem_workspace import SmemWorkspace
from ......helpers.utils import ceil_div, round_up
from ......quant_def import CombineFormat, QuantKind
from ......communication.nvlink_domain.token_comm import TokenCommArgs, TokenCommNonDeterministic
from ......communication.nvlink_domain.token_comm_deterministic import TokenCommDeterministic
from ..topk_reduce import TopkReduce
from .glu_mxfp8_fc12_kernel import Sm107Mxfp8GluFc12Kernel


_AB_DTYPE_TO_QUANT_KIND = {
    cutlass.Float8E4M3FN: QuantKind.mxfp8_e4m3,
    cutlass.Float8E5M2: QuantKind.mxfp8_e5m2,
}
_QUANT_KIND_TO_AB_DTYPE = {str(k): d for d, k in _AB_DTYPE_TO_QUANT_KIND.items()}

# TVM-FFI export symbol for the AOT-compiled callable (consumed by ``tester.compiler``).
_aot_symbol_prefix = "rubin_mega_moe_glu_mxfp8_aot"


class Sm107MegaMoEMxfp8GluKernel(Sm107Mxfp8GluFc12Kernel, KernelClass):
    """Multi-rank MegaMoE wrapper around the lean mxfp8 GLU FC12 kernel."""

    fc1_output_region = "rubin.glu_mxfp8.mega.fc1_output"
    fc1_output_sf_region = "rubin.glu_mxfp8.mega.fc1_output_sf"
    fc1_done_counter_region = "rubin.glu_mxfp8.mega.fc1_done_counter"

    # Reserved on top of the exact token_comm/sched SMEM to cover smem.allocate inter-allocation
    # alignment padding that _compute_stages does not model (see _smem_misc_budget_bytes).
    _SMEM_ALLOC_MARGIN = 2048

    @classmethod
    def problem_desc_require(cls):
        return {
            "expert_count": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "hidden_size": StaticOrRuntimeIntegerType,
            "quant_kind": str,
            "combine_format": CombineFormat,
            "world_size": int,
            "local_rank": int,
            "topk": int,
            "max_tokens_per_rank": int,
            "apply_topk_at_fc1": bool,
            "gate_up_clamp": Optional[float],
        }

    @classmethod
    def impl_desc_require(cls):
        return {
            "mma_tiler_mnk": tuple,
            "cluster_shape_mnk": tuple,
            "use_2cta_instrs": bool,
            "group_hint": int,
            "token_padding_block": int,
            "sf_padding_block": int,
            "load_balance_mode": str,
            "force_static_sched": bool,
            "clc_bundle_size": Optional[int],
            "num_sched_stages": Optional[int],
            "acc_dtype": type,
            "sf_vec_size": int,
            "launch_cluster_count": int,
            "fc2_in_kernel_topk_reduce": bool,
            "token_back_mode": str,
            "epi_flag_batch": tuple,
            "flag_batch": int,
            "generate_c": bool,
            "use_stg_fc1": bool,
            "act_func": str,
            "fc2_use_bulk": bool,
            "fc2_tma_stages": OptionalRequirement(int),
        }

    def name(self) -> str:
        return (
            f"sm107_megamoe_glu_{self.quant_kind}_m{self.mma_tiler_mnk[0]}n{self.mma_tiler_mnk[1]}"
            f"k{self.mma_tiler_mnk[2]}_e{self.expert_count}_ep{self.world_size}_topk{self.topk}_"
            f"h{self.hidden_size}_i{self.intermediate_gateup_size}_combine{self.combine_format}_"
            f"tokenback{self.token_back_mode}_hint{self.group_hint}_"
            f"epi{self.epi_flag_batch[0]}x{self.epi_flag_batch[1]}_tif{self.flag_batch}_"
            f"fc2bulk{int(self.fc2_use_bulk)}x{self.fc2_tma_stages}_"
            f"redtopk{int(self.reduce_topk_in_kernel)}"
        )

    def aot_compile(self, out_path: Optional[str] = None, **_compile_kwargs):
        """Compile against fake (metadata-only) inputs; ``out_path=None`` returns the in-memory callable."""
        import math

        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr
        from cutlass.cute.typing import AddressSpace, sym_int64
        from cutlass.cutlass_dsl import Int32, Int64

        from ......communication.nvlink_domain.symmetric_buffer import SymmetricBufferHost

        def fake_tensor(dtype, shape, stride_order, dynamic_axes, alignment):
            extents = tuple(
                sym_int64(divisibility=math.gcd(int(extent), 128)) if axis in dynamic_axes else int(extent)
                for axis, extent in enumerate(shape)
            )
            return make_fake_compact_tensor(dtype, extents, stride_order=stride_order, assumed_align=alignment)

        tokens = self.max_tokens_per_rank
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        experts = self.num_experts_per_rank
        sf_vec_size = self.sf_vec_size
        fc1_weight_sf_columns = round_up(intermediate_gateup, 128) * round_up(hidden // sf_vec_size, 4)
        fc2_weight_sf_columns = round_up(hidden, 128) * round_up(intermediate_downproj // sf_vec_size, 4)
        output_dtype = cutlass.BFloat16
        # Weight SF and activation SF share the E8M0 block-scale dtype for mxfp8.
        weight_sf_dtype = self.token_comm.activation_sf_dtype

        fake_arguments = dict(
            activation=fake_tensor(self.token_comm.activation_dtype, (tokens, hidden), (1, 0), {0}, 16),
            activation_sf=fake_tensor(
                self.token_comm.activation_sf_dtype,
                (tokens, self.token_comm.activation_sf_hidden_padded),
                (1, 0),
                {0},
                16,
            ),
            topk_indices=fake_tensor(cutlass.Int32, (tokens, self.topk), (1, 0), {0}, 16),
            topk_scores=fake_tensor(cutlass.Float32, (tokens, self.topk), (1, 0), {0}, 4),
            fc1_weight=fake_tensor(self.ab_dtype, (experts, hidden, intermediate_gateup), (2, 0, 1), {0, 2}, 16),
            fc1_weight_sf=fake_tensor(weight_sf_dtype, (experts, fc1_weight_sf_columns), (1, 0), {0}, 16),
            fc2_weight=fake_tensor(self.ab_dtype, (experts, intermediate_downproj, hidden), (2, 0, 1), {0, 2}, 16),
            fc2_weight_sf=fake_tensor(weight_sf_dtype, (experts, fc2_weight_sf_columns), (1, 0), {0}, 16),
            output_activation=fake_tensor(output_dtype, (tokens, hidden), (1, 0), {0}, 16),
            overflow_flag=fake_tensor(cutlass.Int32, (1,), (0,), set(), 4),
            local_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            shared_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            peer_rank_ptr_mapper_host=SymmetricBufferHost(
                base_address=Int64(0),
                offsets=tuple(Int64(0) for _ in range(self.world_size)),
                rank=Int32(0),
                max_ranks=self.world_size,
            ),
            stream=make_fake_stream(),
        )
        if self.generate_c:
            fake_arguments["fc1_c"] = fake_tensor(output_dtype, (tokens, intermediate_gateup), (1, 0), {0}, 16)
        else:
            fake_arguments["fc1_c"] = None

        compiled = cute.compile[cute.EnableTVMFFI(True)](self, **fake_arguments)
        if out_path is None:
            return compiled
        compiled.export_to_c(out_path, function_name=_aot_symbol_prefix, export_only_tvm_ffi_symbols=True)
        return out_path

    @staticmethod
    def load_compiled(path: str):
        from cutlass.cute.runtime import load_module

        return load_module(path, enable_tvm_ffi=True)[_aot_symbol_prefix]

    @classmethod
    def from_kwargs(
        cls,
        # Base-class (lean FC12) kwargs.
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
        sf_vec_size: int = 32,
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        max_tokens_per_rank: int,
        hidden: int,
        launch_cluster_count: int,
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_mode: Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"] = "epi_warps",
        epi_flag_batch: Optional[Tuple[int, int]] = (4, 2),
        flag_batch: int = 1,
        gate_up_clamp: Optional[float] = None,
        apply_topk_in_fc1: bool = True,
        generate_c: bool = False,
        use_stg_fc1: bool = False,
        combine_format: Optional[CombineFormat] = None,
        act_func: str = "swiglu",
        fc2_use_bulk: bool = False,
        fc2_tma_stages: Optional[int] = None,
    ) -> "Sm107MegaMoEMxfp8GluKernel":
        """Build the ``(ProblemDesc, ImplDesc)`` pair from the legacy flat signature."""
        if static_expert_shape is None:
            raise NotImplementedError("Sm107MegaMoEMxfp8GluKernel requires a static_expert_shape.")
        if hidden != static_expert_shape[2]:
            raise ValueError(f"hidden ({hidden}) must equal static_expert_shape[2] ({static_expert_shape[2]}).")
        if ab_dtype not in _AB_DTYPE_TO_QUANT_KIND:
            raise ValueError(f"ab_dtype {ab_dtype} has no mxfp8 QuantKind.")
        num_experts_per_rank, intermediate_gateup, _hidden = static_expert_shape
        combine_format = CombineFormat.parse("bf16" if combine_format is None else str(combine_format))
        problem_desc = ProblemDesc(
            {
                "expert_count": world_size * num_experts_per_rank,
                "intermediate_gateup_size": intermediate_gateup,
                "hidden_size": hidden,
                "quant_kind": str(_AB_DTYPE_TO_QUANT_KIND[ab_dtype]),
                "combine_format": combine_format,
                "world_size": world_size,
                "local_rank": local_rank,
                "topk": num_topk,
                "max_tokens_per_rank": max_tokens_per_rank,
                "apply_topk_at_fc1": apply_topk_in_fc1,
                "gate_up_clamp": gate_up_clamp,
            }
        )
        impl_desc = ImplDesc(
            {
                "mma_tiler_mnk": tuple(mma_tiler_mnk),
                "cluster_shape_mnk": tuple(cluster_shape_mnk),
                "use_2cta_instrs": use_2cta_instrs,
                "group_hint": group_hint,
                "token_padding_block": token_padding_block,
                "sf_padding_block": sf_padding_block,
                "load_balance_mode": load_balance_mode,
                "force_static_sched": force_static_sched,
                "clc_bundle_size": clc_bundle_size,
                "num_sched_stages": num_sched_stages,
                "acc_dtype": acc_dtype,
                "sf_vec_size": sf_vec_size,
                "launch_cluster_count": launch_cluster_count,
                "fc2_in_kernel_topk_reduce": fc2_in_kernel_topk_reduce,
                "token_back_mode": token_back_mode,
                "epi_flag_batch": tuple(epi_flag_batch) if epi_flag_batch is not None else (1, 1),
                "flag_batch": flag_batch,
                "generate_c": generate_c,
                "use_stg_fc1": use_stg_fc1,
                "act_func": act_func,
                "fc2_use_bulk": fc2_use_bulk,
                # OptionalRequirement: present only when set (absent == None).
                **({"fc2_tma_stages": fc2_tma_stages} if fc2_tma_stages is not None else {}),
            }
        )
        return cls(problem_desc, impl_desc)

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        # -- Extract descriptors into locals matching the legacy param names so the body below
        #    is unchanged; derive the base-class flat inputs (static_expert_shape, ab_dtype). --
        world_size = problem_desc["world_size"]
        local_rank = problem_desc["local_rank"]
        num_topk = problem_desc["topk"]
        max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        hidden = problem_desc["hidden_size"]
        gate_up_clamp = problem_desc["gate_up_clamp"]
        apply_topk_in_fc1 = problem_desc["apply_topk_at_fc1"]
        combine_format = problem_desc["combine_format"]
        _quant_kind = problem_desc["quant_kind"]
        ab_dtype = _QUANT_KIND_TO_AB_DTYPE[_quant_kind]
        static_expert_shape = (
            problem_desc["expert_count"] // world_size,
            problem_desc["intermediate_gateup_size"],
            hidden,
        )

        mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        cluster_shape_mnk = impl_desc["cluster_shape_mnk"]
        use_2cta_instrs = impl_desc["use_2cta_instrs"]
        group_hint = impl_desc["group_hint"]
        token_padding_block = impl_desc["token_padding_block"]
        sf_padding_block = impl_desc["sf_padding_block"]
        load_balance_mode = impl_desc["load_balance_mode"]
        force_static_sched = impl_desc["force_static_sched"]
        clc_bundle_size = impl_desc["clc_bundle_size"]
        num_sched_stages = impl_desc["num_sched_stages"]
        acc_dtype = impl_desc["acc_dtype"]
        sf_vec_size = impl_desc["sf_vec_size"]
        launch_cluster_count = impl_desc["launch_cluster_count"]
        fc2_in_kernel_topk_reduce = impl_desc["fc2_in_kernel_topk_reduce"]
        token_back_mode = impl_desc["token_back_mode"]
        epi_flag_batch = impl_desc["epi_flag_batch"]
        flag_batch = impl_desc["flag_batch"]
        generate_c = impl_desc["generate_c"]
        use_stg_fc1 = impl_desc["use_stg_fc1"]
        act_func = impl_desc["act_func"]
        fc2_use_bulk = impl_desc["fc2_use_bulk"]
        fc2_tma_stages = impl_desc.get("fc2_tma_stages")

        if static_expert_shape is None:
            raise NotImplementedError("Sm107MegaMoEMxfp8GluKernel requires a static_expert_shape.")
        if hidden != static_expert_shape[2]:
            raise ValueError(f"hidden ({hidden}) must equal static_expert_shape[2] ({static_expert_shape[2]}).")
        token_back_by_dispatch = token_back_mode != "epi_warps"
    
        combine_format = CombineFormat.parse("bf16" if combine_format is None else str(combine_format))
        if fc2_in_kernel_topk_reduce and (token_back_by_dispatch or combine_format.is_quantized):
            raise ValueError("fc2_in_kernel_topk_reduce requires epi_warps + non-quantized (bf16) combine.")
        if fc2_in_kernel_topk_reduce and not apply_topk_in_fc1:
            raise ValueError("fc2_in_kernel_topk_reduce requires apply_topk_in_fc1=True.")
        if token_back_mode not in ("epi_warps", "standalone_warps", "reuse_dispatch_warps"):
            raise ValueError(f"unsupported token_back_mode={token_back_mode!r}.")
        if ab_dtype not in _AB_DTYPE_TO_QUANT_KIND:
            raise ValueError(f"ab_dtype {ab_dtype} has no mxfp8 QuantKind.")
        # FC2 bulk store
        if fc2_use_bulk and not combine_format.is_quantized:
            raise ValueError("fc2_use_bulk currently supports only a quantized (mxfp8) combine format.")
        if fc2_tma_stages is not None and not fc2_use_bulk:
            raise ValueError("fc2_tma_stages requires fc2_use_bulk=True.")

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
            fc2_in_kernel_topk_reduce=fc2_in_kernel_topk_reduce,
            token_back_by_dispatch=token_back_by_dispatch,
            epi_flag_batch=epi_flag_batch,
            gate_up_clamp=gate_up_clamp,
            apply_topk_in_fc1=apply_topk_in_fc1,
            generate_c=generate_c,
            use_stg_fc1=use_stg_fc1,
            act_func=act_func,
            fc2_use_bulk=fc2_use_bulk,
            fc2_tma_stages=fc2_tma_stages,
        )

        # --- Warp topology: expand to 12 warps (or 16 for standalone token-back). ---
        self.enable_token_comm = True
        self.dispatch_warp_id = (8, 9, 10, 11)
        self.token_back_mode = token_back_mode
        self.token_back_standalone = token_back_by_dispatch and token_back_mode == "standalone_warps"
        self.token_back_warp_id = (12, 13, 14, 15) if self.token_back_standalone else None
        num_token_back_warps = len(self.token_back_warp_id) if self.token_back_standalone else 0
        self.threads_per_cta = 32 * (
            len(self.epilogue_warp_id) + 4 + len(self.dispatch_warp_id) + num_token_back_warps
        )

        # --- MegaMoE constants. ---
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden
        self.launch_cluster_count = launch_cluster_count
        self.combine_format = combine_format
        self.num_experts_per_rank = static_expert_shape[0]
        self.intermediate_gateup = static_expert_shape[1]
        self.intermediate_downproj = self.intermediate_gateup // 2
        self.num_total_experts = world_size * self.num_experts_per_rank
        self.reduce_topk_in_kernel = fc2_in_kernel_topk_reduce
        self.token_back_schedule_mode = load_balance_mode if load_balance_mode == "atomic_counter" else "static"

        # --- next Router-push token communication component. ---
        mma_cta_count = 2 if use_2cta_instrs else 1
        cta_tile_m = mma_tiler_mnk[0] // mma_cta_count
        cluster_m, cluster_n = self.cluster_shape_mn
        tokens_per_fc1_ready_slot = cta_tile_m * cluster_m
        hidden_per_fc2_cluster_tile = cta_tile_m * cluster_m
        fc2_done_signals_per_token_tile = ceil_div(hidden, hidden_per_fc2_cluster_tile) * cluster_m * cluster_n
        promised_launchable_sm_count = launch_cluster_count * cluster_m * cluster_n
        quant_kind = _AB_DTYPE_TO_QUANT_KIND[ab_dtype]
        tc_problem_desc = ProblemDesc(
            {
                "world_size": world_size,
                "expert_count": self.num_total_experts,
                "topk": num_topk,
                "max_tokens_per_rank": max_tokens_per_rank,
                "hidden_size": hidden,
                "quant_kind": str(quant_kind),
                "combine_format": combine_format,
                "apply_topk_at_fc1": apply_topk_in_fc1,
            }
        )
        tc_impl_desc = ImplDesc(
            {
                "token_padding_block": token_padding_block,
                "sf_padding_block": sf_padding_block,
                "tokens_per_fc1_ready_slot": tokens_per_fc1_ready_slot,
                "fc2_done_signals_per_token_tile": fc2_done_signals_per_token_tile,
                "promised_launchable_sm_count": promised_launchable_sm_count,
                "token_in_flag_batch": flag_batch,
                "token_back_mode": token_back_mode,
                "token_back_schedule_mode": self.token_back_schedule_mode,
                "reduce_topk_in_kernel": fc2_in_kernel_topk_reduce,
            }
        )
        self.token_comm = TokenCommNonDeterministic(tc_problem_desc, tc_impl_desc)
        self.pool_token_capacity = self.token_comm.worst_case_token_count

        # --- SMEM sub-buffer for the token_comm transport (allocated in the device kernel). ---
        tc_smem_ws = SmemWorkspace()
        self.token_comm.register_smem_regions(tc_smem_ws)
        tc_smem_ws.finalize(max_bytes=self.smem_capacity)
        self.tc_smem_ws = tc_smem_ws
        self._token_comm_smem_bytes = tc_smem_ws.total_bytes

        # Build the scheduler
        _ec, _ig, _hd = static_expert_shape
        self._build_scheduler(
            expert_cnt=_ec,
            intermediate_gateup=_ig,
            hidden_dim=_hd,
            launch_cluster_count=launch_cluster_count,
        )
        self._sched_smem_bytes = self.sched_smem_ws.total_bytes

        # --- Post-kernel top-k reduction (skipped under in-kernel REDG reduce). ---
        self._topk_reduce = (
            None if fc2_in_kernel_topk_reduce else TopkReduce(hidden, num_topk, combine_format)
        )

        # --- Device workspace (next model): fc1 pool/output + token_comm regions. ---
        self._mega_device_workspace = self._build_megamoe_device_workspace()

        self.expert_count = self.num_total_experts
        self.intermediate_gateup_size = self.intermediate_gateup
        self.hidden_size = hidden
        self.quant_kind = _quant_kind
        self.topk = num_topk
        self.apply_topk_at_fc1 = apply_topk_in_fc1
        self.cluster_shape_mnk = tuple(cluster_shape_mnk)
        self.mma_tiler_mnk = tuple(mma_tiler_mnk)
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.load_balance_mode = load_balance_mode
        self.force_static_sched = force_static_sched
        self.clc_bundle_size = clc_bundle_size
        self.num_sched_stages = num_sched_stages
        self.acc_dtype = acc_dtype
        self.sf_vec_size = sf_vec_size
        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.epi_flag_batch = tuple(epi_flag_batch)
        self.flag_batch = flag_batch
        self.generate_c = generate_c
        self.use_stg_fc1 = use_stg_fc1
        self.act_func = act_func
        self.use_2cta_instrs = use_2cta_instrs
        self.gate_up_clamp = gate_up_clamp

    def _smem_misc_budget_bytes(self) -> int:
        """Reserve the token_comm transport SMEM on top of the base misc budget."""
        _sched = getattr(self, "_sched_smem_bytes", 0)
        return (
            super()._smem_misc_budget_bytes()
            + self._token_comm_smem_bytes
            + _sched
            + self._SMEM_ALLOC_MARGIN
        )

    def _build_megamoe_device_workspace(self) -> DeviceWorkspace:
        """Register the FC1 output/pool + fc1_done_counter + all token_comm regions."""
        sf_dtype = cutlass.Float8E8M0FNU
        sf_column_count = round_up(ceil_div(self.intermediate_downproj, self.sf_vec_size), 4)
        max_sf_rows = self.token_comm.worst_case_sf_token_count
        counter_slot_count = self.token_comm.max_fc1_ready_slot_count

        device_workspace = DeviceWorkspace()
        device_workspace.register(
            self.fc1_output_region,
            self.ab_dtype,
            (self.pool_token_capacity, self.intermediate_downproj),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        device_workspace.register(
            self.fc1_output_sf_region,
            sf_dtype,
            (max_sf_rows, sf_column_count),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        device_workspace.register(
            self.fc1_done_counter_region,
            cutlass.Int32,
            (counter_slot_count,),
            buffer_space="local",
            byte_alignment=16,
            reset="tail_reset",
        )
        self.token_comm.register_device_workspace(device_workspace)
        device_workspace.finalize()
        return device_workspace

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return required (local, shared/symmetric) workspace bytes."""
        return self._mega_device_workspace.local_and_shared_bytes

    @property
    def require_zero_workspace_leading_bytes(self) -> Tuple[int, int]:
        return self._mega_device_workspace.require_zero_workspace_leading_bytes

    # =========================================================================
    # token_comm_hook_* -- the lean device method's integration seams, here
    # filled with next's Router-push TokenCommDeterministic calls.
    # =========================================================================

    def token_comm_extra_smem_storage_class(self) -> type:
        """SMEM struct for the token_comm transport overlay (allocated in the device kernel)."""
        return self.tc_smem_ws.storage_class()

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        """Pointer the FC1 scheduler/extension spins on; token_in increments it per ready slot."""
        return self.token_comm.fc1_ready_counter_pointer(self._mega_device_workspace)

    def sched_ext_fc1_peek_threshold(self) -> int:  # noqa: D401 - lean hook override point
        return super().sched_ext_fc1_peek_threshold()

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        """The scheduler warp must wait for the Router to publish per-expert sizes."""
        self.token_comm.wait_for_sizes_ready(self._mega_device_workspace)

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(self, token_comm_args, work_tile_info):
        """No-op: FC1 input readiness is enforced by the scheduler extension's fc1_ready spin."""
        pass

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self, token_comm_args, token_comm_storage, *, warp_idx, lane_idx, tidx
    ):
        """Transfer warps (8-11): pull activation from peers into the local FC1 pool."""
        self.token_comm.token_in(self.tc_smem_ws, token_comm_storage.buffer.data_ptr())
        if cutlass.const_expr(self.token_comm.token_back_enabled and not self.token_back_standalone):
            self.token_comm.token_back(self.tc_smem_ws, token_comm_storage.buffer.data_ptr())

    @cute.jit
    def token_comm_hook_token_back_warp_body(
        self, token_comm_args, token_comm_storage, *, warp_idx, lane_idx, tidx
    ):
        """Standalone token-back warps (12-15): push FC2 output back to source ranks."""
        self.token_comm.token_back(self.tc_smem_ws, token_comm_storage.buffer.data_ptr())

    @cute.jit
    def token_comm_hook_tail_reset_shared_counters(self, token_comm_args, *, warp_idx, lane_idx, tidx):
        """Absorbed into reset_tail (kernel_tail hook)."""
        pass

    @cute.jit
    def token_comm_hook_kernel_tail(self, token_comm_args, *, warp_idx, lane_idx, tidx):
        """Cross-rank drain + workspace tail reset, performed by the transfer warps.

        The whole-CTA barrier is REQUIRED (mirrors inference's mainloop kernel-tail
        sync_threads): it forces every compute warp (scheduler / tma / mma / epilogue) to
        finish its consume loop -- including any ``ext.wait_for_input`` spin on ``fc1_ready``
        -- before the transfer warps run ``reset_tail``.  ``reset_tail`` tail-resets the
        (local, GPU-wide) ``fc1_ready`` / ``fc2_done`` counters; without this barrier a fast
        CTA's transfer warps can zero a counter that a slower compute warp is still spinning
        on, which non-deterministically deadlocks (~1/5 runs).
        """
        cute.arch.sync_threads()
        # reset_tail must run on EXACTLY the 4 transfer/token_in warps (8-11).  For
        # standalone_warps (16 warps) the token_back warps (12-15) must be EXCLUDED: their
        # thread_idx aliases (thread_idx % transfer_thread_count) back onto transfer warps 0-3,
        # so including them double-counts the reset_tail NVLink grid barrier -> deadlock.
        if (warp_idx >= self.dispatch_warp_id[0]) & (warp_idx <= self.dispatch_warp_id[-1]):
            self.token_comm.reset_tail()
        self.token_comm.remove_device_members()

    # =========================================================================
    # Host launch: Router kernel -> fused MegaMoE main kernel -> top-k reduction.
    # =========================================================================

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,          # (max_tokens_per_rank, hidden) raw per-rank, symmetric heap
        activation_sf: cute.Tensor,       # (max_tokens_per_rank, hidden // sf_vec_size), symmetric
        topk_indices: cute.Tensor,        # (max_tokens_per_rank, topk)
        topk_scores: cute.Tensor,         # (max_tokens_per_rank, topk) Float32
        fc1_weight: cute.Tensor,          # (experts_per_rank, hidden, intermediate_gateup)
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,          # (experts_per_rank, intermediate_downproj, hidden)
        fc2_weight_sf: cute.Tensor,
        output_activation: cute.Tensor,   # (max_tokens_per_rank, topk, hidden) final combined output
        fc1_c: Optional[cute.Tensor],
        overflow_flag: cute.Tensor,       # (1,) Int32, per-rank FC12 overflow output
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,   # symmetric (NVLink) heap base
        peer_rank_ptr_mapper_host,
        stream: cuda.CUstream,
    ) -> None:
        """Launch the Router, then the fused main kernel, then (optionally) the top-k reduce."""
        dw = self._mega_device_workspace
        local_rank = peer_rank_ptr_mapper_host.rank
        router_scores = topk_scores if cutlass.const_expr(self.apply_topk_in_fc1) else None
        self.token_comm.launch_router(
            topk_indices,
            router_scores,
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper_host,
            dw,
            stream,
        )
        peer_mapper = peer_rank_ptr_mapper_host.make_device_object()
        dw.assign_device_members(local_workspace, shared_workspace)

        activation_pool = self.token_comm.fc1_activation_tensor(dw)
        _sf_pool_atom = self.token_comm.fc1_activation_sf_tensor(dw)
        activation_sf_pool = cute.make_tensor(
            _sf_pool_atom.iterator,
            cute.make_layout(
                (self.token_comm.worst_case_sf_token_count, self.hidden // self.sf_vec_size),
                stride=(self.token_comm.activation_sf_hidden_padded, 1),
            ),
        )
        fc1_output = dw.tensor(self.fc1_output_region)
        fc1_output_sf = dw.tensor(self.fc1_output_sf_region)
        fc1_done_counter = dw.tensor(self.fc1_done_counter_region)
        pool_topk_scores = self.token_comm.fc1_topk_scores_tensor(dw)

        if cutlass.const_expr(self.reduce_topk_in_kernel):
            # In-kernel top-k reduce (epi_warps + bf16 combine)
            pre_reduced = cute.make_tensor(
                output_activation.iterator,
                cute.make_layout(
                    (output_activation.shape[0], 1, output_activation.shape[1]),
                    stride=(
                        output_activation.stride[0],
                        output_activation.stride[0],
                        output_activation.stride[1],
                    ),
                ),
            )
            pre_reduced_sf = None
        else:
            pre_reduced = self.token_comm.pre_reduced_activation_tensor(dw)
            pre_reduced_sf = self.token_comm.pre_reduced_activation_sf_tensor(dw)

        if cutlass.const_expr(self.token_comm.token_back_push_data):
            # token_back-by-dispatch (standalone_warps / reuse_dispatch_warps)
            fc2_output = self.token_comm.fc2_activation_tensor(dw)
        else:
            # epi_warps: the epilogue peer-writes FC2 directly.
            _combine_hidden = pre_reduced.shape[2]
            fc2_output = cute.make_tensor(
                pre_reduced.iterator,
                cute.make_layout(
                    (pre_reduced.shape[0] * pre_reduced.shape[1], _combine_hidden),
                    stride=(_combine_hidden, 1),
                ),
            )

        super().__call__(
            activation_pool,
            fc1_weight,
            activation_sf_pool,
            fc1_weight_sf,
            fc1_output,
            fc1_output_sf,
            fc2_weight,
            fc2_weight_sf,
            fc2_output,
            pool_topk_scores,
            fc1_done_counter,
            offs=None,
            max_active_clusters=self.launch_cluster_count,
            stream=stream,
            fc1_c=fc1_c,
            overflow_flag=overflow_flag,
            mega_peer_rank_ptr_mapper=peer_mapper,
            mega_local_rank=local_rank,
            mega_local_workspace=local_workspace,
            mega_shared_workspace=shared_workspace,
            mega_activation=activation,
            mega_activation_sf=activation_sf,
            mega_pre_reduced_activation=pre_reduced,
            mega_pre_reduced_activation_sf=pre_reduced_sf,
        )

        # Post-kernel top-k reduction: dequant + FMA(topk score) into the final output.
        if cutlass.const_expr(not self.reduce_topk_in_kernel):
            reduce_scores = None if cutlass.const_expr(self.apply_topk_in_fc1) else topk_scores
            self._topk_reduce(pre_reduced, pre_reduced_sf, output_activation, reduce_scores, stream)
