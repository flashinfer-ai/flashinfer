"""One-launch mixed-bitrate ``trellis_t256`` MoE execution.

The route packer assigns every global expert to one combined expert namespace.
Input/intermediate rotations therefore run once. Per-tile dispatch resolves the
combined expert to a bitrate-specialized K3, K4, or K5 decoder while preserving
the single cooperative FC1/activation/FC2 grid used by homogeneous trellis
execution. The decoder codebook is a compile-time parameter shared with the
fused W4A16 kernel ABI.

The module stays internal because checkpoint interpretation and runtime planning
belong to the serving framework; B12X owns only the prepared kernel path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Protocol, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import OptLevel
from cutlass.cutlass_dsl import Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import get_ptr_as_int64, shared_ptr_to_u32
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from .host import (
    max_packed_route_slots,
    packed_gemm_scratch_elements,
    route_pack_warmup_token_counts,
)
from .kernel import (
    _LUT_E4M3_VALUE_TABLE_ENTRIES,
    _LUT_E4M3_SMEM_REGION_BYTES,
    W4A16FusedMoeKernel,
    _cutlass_element_dtype,
    _fake_m_for_specialization,
    _trellis256_execution_lut,
    compile_w4a16_topk_sum,
    pack_topk_routes_by_expert,
)


@dataclass(frozen=True)
class MixedTrellisCompileResult:
    compiled: object
    topk_sum: object
    trellis_lut: torch.Tensor
    size_m: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    tier0_num_experts: int
    tier1_num_experts: int
    tier0_bits: int
    tier1_bits: int
    trellis_codebook: str
    fc1_tile_k: int
    fc1_tile_n: int
    fc2_tile_k: int
    fc2_tile_n: int
    moe_block_size: int
    fc2_moe_block_size: int
    fc2_schedule_route_block_factor: int
    max_m_blocks: int
    blocks_per_sm: int
    sms: int
    shared_memory_bytes: int
    rotation_input_dtype: str
    route_ids_dtype: torch.dtype
    broadcast_suh: bool
    broadcast_svh: bool


@dataclass(frozen=True)
class MixedTrellis3CompileResult(MixedTrellisCompileResult):
    """Launch metadata for the K3/K4/K5 three-tier specialization."""

    tier2_num_experts: int
    tier2_bits: int


@dataclass(frozen=True)
class MixedTrellisRotations:
    intermediate: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor


@dataclass(frozen=True)
class MixedTrellisBinding:
    """Validated fixed artifacts for a two-tier launch."""

    launch: MixedTrellisCompileResult
    tiers: tuple[MixedTrellisTier, MixedTrellisTier]
    global_to_combined: torch.Tensor
    descriptor_map: torch.Tensor
    rotations: MixedTrellisRotations
    gate_counts: tuple[int, int]
    up_counts: tuple[int, int]
    fc2_counts: tuple[int, int]
    total_experts: int
    route_num_experts: int
    device: torch.device
    tier_pointers: tuple[tuple[object, ...], ...]
    descriptor_map_ptr: object
    intermediate_rotations_ptr: object
    gate_suh_ptr: object
    up_suh_ptr: object
    global_to_combined_ptr: object
    down_svh_ptr: object


@dataclass(frozen=True)
class MixedTrellis3Binding:
    """Validated fixed artifacts for a three-tier launch."""

    launch: MixedTrellis3CompileResult
    tiers: tuple[MixedTrellisTier, MixedTrellisTier, MixedTrellisTier]
    global_to_combined: torch.Tensor
    descriptor_map: torch.Tensor
    rotations: MixedTrellisRotations
    gate_counts: tuple[int, int, int]
    up_counts: tuple[int, int, int]
    fc2_counts: tuple[int, int, int]
    total_experts: int
    route_num_experts: int
    device: torch.device
    tier_pointers: tuple[tuple[object, ...], ...]
    descriptor_map_ptr: object
    intermediate_rotations_ptr: object
    gate_suh_ptr: object
    up_suh_ptr: object
    global_to_combined_ptr: object
    down_svh_ptr: object


@dataclass
class MixedTrellisBuffers:
    rotation_gate: torch.Tensor
    rotation_up: torch.Tensor
    fc1: torch.Tensor
    activated: torch.Tensor
    fc2: torch.Tensor
    output: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor
    fc1_scratch: torch.Tensor
    fc2_scratch: torch.Tensor
    workspace: torch.Tensor


class MixedTrellisTier(Protocol):
    """Prepared Trellis tier fields consumed by the mixed launch."""

    num_experts: int
    trellis_codebook: str
    intermediate_rotations: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2_global_scale: torch.Tensor


class W4A16MixedTrellisKernel:
    """One cooperative grid over two native Trellis bitrates."""

    # Persistent compile keys do not include launch source text. Change this
    # version whenever the compiled argument or tensor-layout contract changes.
    ABI_VERSION = 13

    def __init__(
        self,
        *,
        driver: W4A16FusedMoeKernel,
        tier0: W4A16FusedMoeKernel,
        tier1: W4A16FusedMoeKernel,
    ):
        for name, moe in (("driver", driver), ("tier0", tier0), ("tier1", tier1)):
            if not moe.full_rotation or not moe.intermediate_rotation:
                raise ValueError(f"mixed Trellis {name} requires full rotation")
            if moe.direct_topk_routes or moe.tc_decode_fused_sum:
                raise ValueError(f"mixed Trellis {name} requires route packing")
            if moe.weight_layout != "trellis_t256":
                raise ValueError(f"mixed Trellis {name} requires native t256 weights")
            if moe.element_dtype != "fp16":
                raise ValueError(f"mixed Trellis {name} requires fp16 GEMM operands")
        for attr in (
            "size_m",
            "hidden_size",
            "intermediate_size",
            "fc1_cols",
            "top_k",
            "moe_block_size",
            "activation",
            "rotation_input_dtype",
            "broadcast_suh",
            "trellis_codebook",
            "cta_threads",
            "sms",
        ):
            values = tuple(getattr(moe, attr) for moe in (driver, tier0, tier1))
            if values[1:] != values[:-1]:
                raise ValueError(f"mixed Trellis kernels disagree on {attr}: {values}")
        for phase in ("fc1", "fc2"):
            gemms = tuple(getattr(moe, phase) for moe in (driver, tier0, tier1))
            geometry = tuple(
                (
                    g.n_tiles,
                    g.k_tiles,
                    g.tile_n,
                    g.tile_k,
                    g.cta_threads,
                    g.moe_block_size,
                    g.schedule_route_block_factor,
                )
                for g in gemms
            )
            if geometry[1:] != geometry[:-1]:
                raise ValueError(
                    f"mixed Trellis kernels disagree on {phase} geometry: {geometry}"
                )
        fc2_factor = int(driver.fc2.schedule_route_block_factor)
        expected_factor = int(driver.moe_block_size // driver.fc2.moe_block_size)
        if fc2_factor < 1 or expected_factor % fc2_factor != 0:
            raise ValueError(
                "mixed Trellis FC2 schedule factor must divide one packed "
                f"route block: factor={fc2_factor}, maximum={expected_factor}"
            )
        if tier0.num_experts > 256 or tier1.num_experts > 256:
            raise ValueError("tier-local expert ids must fit in eight bits")
        if driver.num_experts != tier0.num_experts + tier1.num_experts:
            raise ValueError("driver expert count must equal the sum of both tiers")
        self.driver = driver
        self.tier0 = tier0
        self.tier1 = tier1
        self.size_m = driver.size_m
        self.hidden_size = driver.hidden_size
        self.intermediate_size = driver.intermediate_size
        self.top_k = driver.top_k
        self.cta_threads = driver.cta_threads
        self.sms = driver.sms
        self.blocks_per_sm = min(
            driver.blocks_per_sm, tier0.blocks_per_sm, tier1.blocks_per_sm
        )
        self.shared_words = max(
            driver.shared_words, tier0.shared_words, tier1.shared_words
        )
        # Each bitrate has a different GEMM scratch footprint. The shared value
        # table must follow the largest pre-LUT region, otherwise a wider tier
        # can overwrite a table placed at the driver's (K3) offset.
        self.lut_e4m3_smem_off = max(
            driver.lut_e4m3_smem_off,
            tier0.lut_e4m3_smem_off,
            tier1.lut_e4m3_smem_off,
        )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            "w4a16_mixed_trellis",
            self.ABI_VERSION,
            self.driver.__cache_key__,
            self.tier0.__cache_key__,
            self.tier1.__cache_key__,
            # Expert counts are runtime artifact data. All E-sized views and
            # dispatch bounds are reconstructed from the launch scalars.
            self.blocks_per_sm,
            self.shared_words,
        )

    @cute.jit
    def _dispatch_tier_gemm(
        self,
        gemm,
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        b_flat: cute.Tensor,
        c_flat: cute.Tensor,
        scales_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights: cute.Tensor,
        c_tmp: cute.Tensor,
        locks: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        local_expert: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
    ):
        factor = gemm.schedule_route_block_factor
        first_route_block = route_block_idx * Int32(factor)
        first_lock_slot = lock_slot * Int32(factor)
        for subtile in cutlass.range_constexpr(factor):
            gemm._run_tile(
                a_flat,
                a_alt_flat,
                b_flat,
                c_flat,
                scales_flat,
                global_scale,
                packed_route_indices,
                topk_weights,
                c_tmp,
                locks,
                trellis_lut_addr,
                smem_base,
                tid,
                first_route_block + Int32(subtile),
                local_expert,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                first_lock_slot + Int32(subtile),
                active_size_m,
            )

    @cute.jit
    def _emit_tier_tile(
        self,
        is_fc1: cutlass.Constexpr,
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        t0_b_flat: cute.Tensor,
        t0_scales_flat: cute.Tensor,
        t0_global_scale: cute.Tensor,
        t1_b_flat: cute.Tensor,
        t1_scales_flat: cute.Tensor,
        t1_global_scale: cute.Tensor,
        c_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights: cute.Tensor,
        c_tmp: cute.Tensor,
        locks: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        active_size_m: Int32,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        tier0_fc2_experts: Int32,
        tier1_fc2_experts: Int32,
        tier0_gate_experts: Int32,
        tier1_gate_experts: Int32,
        tier0_up_experts: Int32,
        tier1_up_experts: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        metadata_block_idx = route_block_idx
        if cutlass.const_expr(not is_fc1):
            metadata_block_idx = route_block_idx // Int32(
                self.driver.moe_block_size
                // (
                    self.driver.fc2.moe_block_size
                    * self.driver.fc2.schedule_route_block_factor
                )
            )
        combined_expert = block_expert_ids[metadata_block_idx].to(Int32)
        total_experts = tier0_num_experts + tier1_num_experts
        # Gate and up projections may assign one expert to different tiers, so
        # FC1 selects a descriptor row from its output-N half. FC2 always uses
        # the down-projection row. The projection-aware tile policy keeps the
        # FC1 half boundary aligned to whole CTA N tiles.
        descriptor_row = Int32(2)
        if cutlass.const_expr(is_fc1):
            fc1_half_tiles = Int32(self.driver.fc1.n_tiles // 2)
            descriptor_row = Int32(0)
            if output_n_tile >= fc1_half_tiles:
                descriptor_row = Int32(1)
        if combined_expert >= Int32(0) and combined_expert < total_experts:
            descriptor = descriptor_map[
                descriptor_row * total_experts + combined_expert
            ].to(Int32)
            if descriptor >= Int32(0):
                tier = descriptor >> Int32(8)
                local_expert = descriptor & Int32(0xFF)
                # FC1 is bounded by the tier's FC1 slot count; FC2 by its own
                # independent count, since per-projection membership lets the
                # two differ. Both remain real bounds.
                if cutlass.const_expr(is_fc1):
                    tier0_in_bounds = local_expert < tier0_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        tier0_in_bounds = local_expert < tier0_up_experts
                else:
                    tier0_in_bounds = local_expert < tier0_fc2_experts
                if tier == Int32(0) and tier0_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier0.fc1
                    else:
                        gemm = self.tier0.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t0_b_flat,
                        c_flat,
                        t0_scales_flat,
                        t0_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )
                if cutlass.const_expr(is_fc1):
                    tier1_in_bounds = local_expert < tier1_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        tier1_in_bounds = local_expert < tier1_up_experts
                else:
                    tier1_in_bounds = local_expert < tier1_fc2_experts
                if tier == Int32(1) and tier1_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier1.fc1
                    else:
                        gemm = self.tier1.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t1_b_flat,
                        c_flat,
                        t1_scales_flat,
                        t1_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

    @cute.jit
    def __call__(
        self,
        rotation_input_ptr: cute.Pointer,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13_ptr: cute.Pointer,
        t0_w2_ptr: cute.Pointer,
        t0_w13_scales_ptr: cute.Pointer,
        t0_w2_scales_ptr: cute.Pointer,
        t0_w13_global_ptr: cute.Pointer,
        t0_w2_global_ptr: cute.Pointer,
        t1_w13_ptr: cute.Pointer,
        t1_w2_ptr: cute.Pointer,
        t1_w13_scales_ptr: cute.Pointer,
        t1_w2_scales_ptr: cute.Pointer,
        t1_w13_global_ptr: cute.Pointer,
        t1_w2_global_ptr: cute.Pointer,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations_ptr: cute.Pointer,
        gate_suh_ptr: cute.Pointer,
        up_suh_ptr: cute.Pointer,
        trellis_lut_ptr: cute.Pointer,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
        # Projection-count scalars form an ABI tail and callers pass them by
        # keyword, so the preceding positional slots remain stable. CuTe DSL
        # traces every parameter and cannot type a None default.
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
    ):
        tier0_experts = cutlass.Int64(tier0_num_experts)
        # FC2 extents are independent of the FC1 slot counts.
        tier0_fc2 = cutlass.Int64(tier0_fc2_experts)
        tier1_fc2 = cutlass.Int64(tier1_fc2_experts)
        # The w13 descriptor is sized by the gate count so the GEMM's
        # cute.size(w13)//2 up-block base lands at gate_count*proj_stride over
        # a tight [gate|up] buffer. When projection counts are omitted, the
        # binding contract requires one gate and one up plane per tier slot.
        tier0_gate = cutlass.Int64(tier0_gate_experts)
        tier1_gate = cutlass.Int64(tier1_gate_experts)
        tier1_experts = cutlass.Int64(tier1_num_experts)
        total_experts = tier0_experts + tier1_experts

        t0_w13 = cute.make_tensor(
            t0_w13_ptr,
            layout=cute.make_layout(
                (
                    tier0_gate
                    * cutlass.Int64(self.hidden_size // 16)
                    * cutlass.Int64(self.driver.fc1_cols // 16)
                    * cutlass.Int64(8 * self.tier0.trellis_bits),
                ),
                stride=(1,),
            ),
        )
        t0_w2 = cute.make_tensor(
            t0_w2_ptr,
            layout=cute.make_layout(
                (
                    tier0_fc2
                    * cutlass.Int64(self.intermediate_size // 16)
                    * cutlass.Int64(self.hidden_size // 16)
                    * cutlass.Int64(8 * self.tier0.trellis_bits),
                ),
                stride=(1,),
            ),
        )
        t1_w13 = cute.make_tensor(
            t1_w13_ptr,
            layout=cute.make_layout(
                (
                    tier1_gate
                    * cutlass.Int64(self.hidden_size // 16)
                    * cutlass.Int64(self.driver.fc1_cols // 16)
                    * cutlass.Int64(8 * self.tier1.trellis_bits),
                ),
                stride=(1,),
            ),
        )
        t1_w2 = cute.make_tensor(
            t1_w2_ptr,
            layout=cute.make_layout(
                (
                    tier1_fc2
                    * cutlass.Int64(self.intermediate_size // 16)
                    * cutlass.Int64(self.hidden_size // 16)
                    * cutlass.Int64(8 * self.tier1.trellis_bits),
                ),
                stride=(1,),
            ),
        )
        t0_w13_scales = cute.make_tensor(
            t0_w13_scales_ptr,
            layout=cute.make_layout(
                (
                    tier0_experts
                    * cutlass.Int64(self.tier0.fc1.scale_k_groups)
                    * cutlass.Int64(self.tier0.fc1.scale_size_n // 4),
                ),
                stride=(1,),
            ),
        )
        t0_w2_scales = cute.make_tensor(
            t0_w2_scales_ptr,
            layout=cute.make_layout(
                (
                    tier0_experts
                    * cutlass.Int64(self.tier0.fc2.scale_k_groups)
                    * cutlass.Int64(self.tier0.fc2.scale_size_n // 4),
                ),
                stride=(1,),
            ),
        )
        t1_w13_scales = cute.make_tensor(
            t1_w13_scales_ptr,
            layout=cute.make_layout(
                (
                    tier1_experts
                    * cutlass.Int64(self.tier1.fc1.scale_k_groups)
                    * cutlass.Int64(self.tier1.fc1.scale_size_n // 4),
                ),
                stride=(1,),
            ),
        )
        t1_w2_scales = cute.make_tensor(
            t1_w2_scales_ptr,
            layout=cute.make_layout(
                (
                    tier1_experts
                    * cutlass.Int64(self.tier1.fc2.scale_k_groups)
                    * cutlass.Int64(self.tier1.fc2.scale_size_n // 4),
                ),
                stride=(1,),
            ),
        )
        t0_w13_global = cute.make_tensor(
            t0_w13_global_ptr,
            layout=cute.make_layout((tier0_experts,), stride=(1,)),
        )
        t0_w2_global = cute.make_tensor(
            t0_w2_global_ptr,
            layout=cute.make_layout((tier0_fc2,), stride=(1,)),
        )
        t1_w13_global = cute.make_tensor(
            t1_w13_global_ptr,
            layout=cute.make_layout((tier1_experts,), stride=(1,)),
        )
        t1_w2_global = cute.make_tensor(
            t1_w2_global_ptr,
            layout=cute.make_layout((tier1_fc2,), stride=(1,)),
        )
        # Descriptor rows are gate, up, and down. A projection-independent
        # partition is represented by three identical rows.
        descriptor_map = cute.make_tensor(
            descriptor_map_ptr,
            layout=cute.make_layout((cutlass.Int64(3) * total_experts,), stride=(1,)),
        )
        intermediate_rotations = cute.make_tensor(
            intermediate_rotations_ptr,
            layout=cute.make_layout(
                (total_experts * cutlass.Int64(3 * self.intermediate_size),),
                stride=(1,),
            ),
        )
        suh_rows = total_experts
        if cutlass.const_expr(self.driver.broadcast_suh):
            suh_rows = cutlass.Int64(1)
        gate_suh = cute.make_tensor(
            gate_suh_ptr,
            layout=cute.make_layout(
                (suh_rows * cutlass.Int64(self.hidden_size),), stride=(1,)
            ),
        )
        up_suh = cute.make_tensor(
            up_suh_ptr,
            layout=cute.make_layout(
                (suh_rows * cutlass.Int64(self.hidden_size),), stride=(1,)
            ),
        )
        trellis_lut = cute.make_tensor(
            trellis_lut_ptr,
            layout=cute.make_layout(
                (Int64(_LUT_E4M3_VALUE_TABLE_ENTRIES),), stride=(1,)
            ),
        )
        rotation_input = cute.make_tensor(
            rotation_input_ptr,
            layout=cute.make_layout(
                (active_m.to(cutlass.Int64) * cutlass.Int64(self.hidden_size),),
                stride=(1,),
            ),
        )
        topk_weights = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout(
                (active_m.to(cutlass.Int64) * cutlass.Int64(self.top_k),),
                stride=(1,),
            ),
        )
        trellis_lut = cute.make_tensor(
            trellis_lut_ptr,
            layout=cute.make_layout(
                (cutlass.Int64(_LUT_E4M3_VALUE_TABLE_ENTRIES),), stride=(1,)
            ),
        )
        self.kernel(
            rotation_input,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            t1_w13,
            t1_w2,
            t1_w13_scales,
            t1_w2_scales,
            t1_w13_global,
            t1_w2_global,
            fc1,
            activated,
            fc2,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            trellis_lut,
            tier0_num_experts,
            tier1_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            cooperative=True,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        rotation_input: cute.Tensor,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_w13_scales: cute.Tensor,
        t0_w2_scales: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t0_w2_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_w13_scales: cute.Tensor,
        t1_w2_scales: cute.Tensor,
        t1_w13_global: cute.Tensor,
        t1_w2_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        trellis_lut: cute.Tensor,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_x_raw)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        trellis_lut_addr = get_ptr_as_int64(trellis_lut, Int32(0))
        phase_lut_addr = trellis_lut_addr
        if cutlass.const_expr(self.driver.lut_e4m3_smem):
            self.driver._lut_smem_copy(
                trellis_lut_addr,
                smem_base + Int32(self.driver.lut_e4m3_smem_off),
                _LUT_E4M3_SMEM_REGION_BYTES,
                tid,
            )
            cute.arch.sync_threads()
            phase_lut_addr = Int64(
                smem_base + Int32(self.driver.lut_e4m3_smem_off)
            )
        fc1_emit = partial(
            self._emit_tier_tile,
            True,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w13_scales,
            t0_w13_global,
            t1_w13,
            t1_w13_scales,
            t1_w13_global,
            fc1,
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            topk_weights,
            fc1_scratch,
            workspace,
            phase_lut_addr,
            smem_base,
            tid,
            active_m,
            tier0_num_experts,
            tier1_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
        )
        fc2_emit = partial(
            self._emit_tier_tile,
            False,
            activated,
            activated,
            t0_w2,
            t0_w2_scales,
            t0_w2_global,
            t1_w2,
            t1_w2_scales,
            t1_w2_global,
            fc2,
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            topk_weights,
            fc2_scratch,
            workspace,
            phase_lut_addr,
            smem_base,
            tid,
            active_m * Int32(self.top_k),
            tier0_num_experts,
            tier1_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
        )
        total_experts = tier0_num_experts + tier1_num_experts
        self.driver._moe_body(
            rotation_gate,
            rotation_up,
            rotation_input,
            t0_w13,
            t0_w2,
            fc1,
            activated,
            fc2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            t0_w13_global,
            Int32(0),
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            descriptor_map,
            trellis_lut_addr,
            trellis_lut_addr,
            total_experts,
            total_experts,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m,
            fc1_emit,
            fc2_emit,
        )


class W4A16MixedTrellis3Kernel(W4A16MixedTrellisKernel):
    """One cooperative grid over three native Trellis bitrates.

    The fixed three-tier ABI keeps K3/K4 execution unchanged and adds K5 only
    for projection-mixed layers whose descriptor map names a third tier.
    """

    # The three-tier ABI has an independent persistent compile namespace.
    ABI_VERSION = 6

    def __init__(
        self,
        *,
        driver: W4A16FusedMoeKernel,
        tier0: W4A16FusedMoeKernel,
        tier1: W4A16FusedMoeKernel,
        tier2: W4A16FusedMoeKernel,
    ):
        kernels = (driver, tier0, tier1, tier2)
        for name, moe in zip(
            ("driver", "tier0", "tier1", "tier2"), kernels, strict=True
        ):
            if not moe.full_rotation or not moe.intermediate_rotation:
                raise ValueError(f"mixed Trellis {name} requires full rotation")
            if moe.direct_topk_routes or moe.tc_decode_fused_sum:
                raise ValueError(f"mixed Trellis {name} requires route packing")
            if moe.weight_layout != "trellis_t256":
                raise ValueError(f"mixed Trellis {name} requires native t256 weights")
            if moe.element_dtype != "fp16":
                raise ValueError(f"mixed Trellis {name} requires fp16 GEMM operands")
        for attr in (
            "size_m",
            "hidden_size",
            "intermediate_size",
            "fc1_cols",
            "top_k",
            "moe_block_size",
            "activation",
            "rotation_input_dtype",
            "broadcast_suh",
            "trellis_codebook",
            "cta_threads",
            "sms",
        ):
            values = tuple(getattr(moe, attr) for moe in kernels)
            if values[1:] != values[:-1]:
                raise ValueError(f"mixed Trellis kernels disagree on {attr}: {values}")
        for phase in ("fc1", "fc2"):
            gemms = tuple(getattr(moe, phase) for moe in kernels)
            geometry = tuple(
                (
                    gemm.n_tiles,
                    gemm.k_tiles,
                    gemm.tile_n,
                    gemm.tile_k,
                    gemm.cta_threads,
                    gemm.moe_block_size,
                    gemm.schedule_route_block_factor,
                )
                for gemm in gemms
            )
            if geometry[1:] != geometry[:-1]:
                raise ValueError(
                    f"mixed Trellis kernels disagree on {phase} geometry: {geometry}"
                )
        fc2_factor = int(driver.fc2.schedule_route_block_factor)
        expected_factor = int(driver.moe_block_size // driver.fc2.moe_block_size)
        if fc2_factor < 1 or expected_factor % fc2_factor != 0:
            raise ValueError(
                "mixed Trellis FC2 schedule factor must divide one packed "
                f"route block: factor={fc2_factor}, maximum={expected_factor}"
            )
        tiers = (tier0, tier1, tier2)
        if any(tier.num_experts > _MAX_TIER_EXPERTS for tier in tiers):
            raise ValueError("tier-local expert ids must fit in eight bits")
        if driver.num_experts != sum(tier.num_experts for tier in tiers):
            raise ValueError("driver expert count must equal the sum of all tiers")
        self.driver = driver
        self.tier0 = tier0
        self.tier1 = tier1
        self.tier2 = tier2
        self.size_m = driver.size_m
        self.hidden_size = driver.hidden_size
        self.intermediate_size = driver.intermediate_size
        self.top_k = driver.top_k
        self.cta_threads = driver.cta_threads
        self.sms = driver.sms
        self.blocks_per_sm = min(tier.blocks_per_sm for tier in kernels)
        self.shared_words = max(tier.shared_words for tier in kernels)
        self.lut_e4m3_smem_off = max(
            tier.lut_e4m3_smem_off for tier in kernels
        )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            "w4a16_mixed_trellis3",
            self.ABI_VERSION,
            self.driver.__cache_key__,
            self.tier0.__cache_key__,
            self.tier1.__cache_key__,
            self.tier2.__cache_key__,
            self.blocks_per_sm,
            self.shared_words,
        )

    @cute.jit
    def _emit_tier_tile3(
        self,
        is_fc1: cutlass.Constexpr,
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        t0_b_flat: cute.Tensor,
        t0_scales_flat: cute.Tensor,
        t0_global_scale: cute.Tensor,
        t1_b_flat: cute.Tensor,
        t1_scales_flat: cute.Tensor,
        t1_global_scale: cute.Tensor,
        t2_b_flat: cute.Tensor,
        t2_scales_flat: cute.Tensor,
        t2_global_scale: cute.Tensor,
        c_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights: cute.Tensor,
        c_tmp: cute.Tensor,
        locks: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        active_size_m: Int32,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        tier2_num_experts: Int32,
        tier0_fc2_experts: Int32,
        tier1_fc2_experts: Int32,
        tier2_fc2_experts: Int32,
        tier0_gate_experts: Int32,
        tier1_gate_experts: Int32,
        tier2_gate_experts: Int32,
        tier0_up_experts: Int32,
        tier1_up_experts: Int32,
        tier2_up_experts: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        metadata_block_idx = route_block_idx
        if cutlass.const_expr(not is_fc1):
            metadata_block_idx = route_block_idx // Int32(
                self.driver.moe_block_size
                // (
                    self.driver.fc2.moe_block_size
                    * self.driver.fc2.schedule_route_block_factor
                )
            )
        combined_expert = block_expert_ids[metadata_block_idx].to(Int32)
        total_experts = tier0_num_experts + tier1_num_experts + tier2_num_experts
        descriptor_row = Int32(2)
        if cutlass.const_expr(is_fc1):
            fc1_half_tiles = Int32(self.driver.fc1.n_tiles // 2)
            descriptor_row = Int32(0)
            if output_n_tile >= fc1_half_tiles:
                descriptor_row = Int32(1)
        if combined_expert >= Int32(0) and combined_expert < total_experts:
            descriptor = descriptor_map[
                descriptor_row * total_experts + combined_expert
            ].to(Int32)
            if descriptor >= Int32(0):
                tier = descriptor >> Int32(8)
                local_expert = descriptor & Int32(0xFF)

                if cutlass.const_expr(is_fc1):
                    t0_in_bounds = local_expert < tier0_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        t0_in_bounds = local_expert < tier0_up_experts
                else:
                    t0_in_bounds = local_expert < tier0_fc2_experts
                if tier == Int32(0) and t0_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier0.fc1
                    else:
                        gemm = self.tier0.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t0_b_flat,
                        c_flat,
                        t0_scales_flat,
                        t0_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

                if cutlass.const_expr(is_fc1):
                    t1_in_bounds = local_expert < tier1_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        t1_in_bounds = local_expert < tier1_up_experts
                else:
                    t1_in_bounds = local_expert < tier1_fc2_experts
                if tier == Int32(1) and t1_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier1.fc1
                    else:
                        gemm = self.tier1.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t1_b_flat,
                        c_flat,
                        t1_scales_flat,
                        t1_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

                if cutlass.const_expr(is_fc1):
                    t2_in_bounds = local_expert < tier2_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        t2_in_bounds = local_expert < tier2_up_experts
                else:
                    t2_in_bounds = local_expert < tier2_fc2_experts
                if tier == Int32(2) and t2_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier2.fc1
                    else:
                        gemm = self.tier2.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t2_b_flat,
                        c_flat,
                        t2_scales_flat,
                        t2_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

    @cute.jit
    def __call__(
        self,
        rotation_input_ptr: cute.Pointer,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13_ptr: cute.Pointer,
        t0_w2_ptr: cute.Pointer,
        t0_w13_scales_ptr: cute.Pointer,
        t0_w2_scales_ptr: cute.Pointer,
        t0_w13_global_ptr: cute.Pointer,
        t0_w2_global_ptr: cute.Pointer,
        t1_w13_ptr: cute.Pointer,
        t1_w2_ptr: cute.Pointer,
        t1_w13_scales_ptr: cute.Pointer,
        t1_w2_scales_ptr: cute.Pointer,
        t1_w13_global_ptr: cute.Pointer,
        t1_w2_global_ptr: cute.Pointer,
        t2_w13_ptr: cute.Pointer,
        t2_w2_ptr: cute.Pointer,
        t2_w13_scales_ptr: cute.Pointer,
        t2_w2_scales_ptr: cute.Pointer,
        t2_w13_global_ptr: cute.Pointer,
        t2_w2_global_ptr: cute.Pointer,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations_ptr: cute.Pointer,
        gate_suh_ptr: cute.Pointer,
        up_suh_ptr: cute.Pointer,
        trellis_lut_ptr: cute.Pointer,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier2_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        tier2_fc2_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier2_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        tier2_up_experts: cutlass.Int32,
    ):
        tier0_experts = cutlass.Int64(tier0_num_experts)
        tier1_experts = cutlass.Int64(tier1_num_experts)
        tier2_experts = cutlass.Int64(tier2_num_experts)
        tier0_fc2 = cutlass.Int64(tier0_fc2_experts)
        tier1_fc2 = cutlass.Int64(tier1_fc2_experts)
        tier2_fc2 = cutlass.Int64(tier2_fc2_experts)
        tier0_gate = cutlass.Int64(tier0_gate_experts)
        tier1_gate = cutlass.Int64(tier1_gate_experts)
        tier2_gate = cutlass.Int64(tier2_gate_experts)
        total_experts = tier0_experts + tier1_experts + tier2_experts

        def weight_tensor(ptr, elements):
            return cute.make_tensor(
                ptr,
                layout=cute.make_layout((elements,), stride=(1,)),
            )

        t0_w13 = weight_tensor(
            t0_w13_ptr,
            tier0_gate
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(self.driver.fc1_cols // 16)
            * cutlass.Int64(8 * self.tier0.trellis_bits),
        )
        t0_w2 = weight_tensor(
            t0_w2_ptr,
            tier0_fc2
            * cutlass.Int64(self.intermediate_size // 16)
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(8 * self.tier0.trellis_bits),
        )
        t1_w13 = weight_tensor(
            t1_w13_ptr,
            tier1_gate
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(self.driver.fc1_cols // 16)
            * cutlass.Int64(8 * self.tier1.trellis_bits),
        )
        t1_w2 = weight_tensor(
            t1_w2_ptr,
            tier1_fc2
            * cutlass.Int64(self.intermediate_size // 16)
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(8 * self.tier1.trellis_bits),
        )
        t2_w13 = weight_tensor(
            t2_w13_ptr,
            tier2_gate
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(self.driver.fc1_cols // 16)
            * cutlass.Int64(8 * self.tier2.trellis_bits),
        )
        t2_w2 = weight_tensor(
            t2_w2_ptr,
            tier2_fc2
            * cutlass.Int64(self.intermediate_size // 16)
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(8 * self.tier2.trellis_bits),
        )

        t0_w13_scales = weight_tensor(
            t0_w13_scales_ptr,
            tier0_experts
            * cutlass.Int64(self.tier0.fc1.scale_k_groups)
            * cutlass.Int64(self.tier0.fc1.scale_size_n // 4),
        )
        t0_w2_scales = weight_tensor(
            t0_w2_scales_ptr,
            tier0_experts
            * cutlass.Int64(self.tier0.fc2.scale_k_groups)
            * cutlass.Int64(self.tier0.fc2.scale_size_n // 4),
        )
        t1_w13_scales = weight_tensor(
            t1_w13_scales_ptr,
            tier1_experts
            * cutlass.Int64(self.tier1.fc1.scale_k_groups)
            * cutlass.Int64(self.tier1.fc1.scale_size_n // 4),
        )
        t1_w2_scales = weight_tensor(
            t1_w2_scales_ptr,
            tier1_experts
            * cutlass.Int64(self.tier1.fc2.scale_k_groups)
            * cutlass.Int64(self.tier1.fc2.scale_size_n // 4),
        )
        t2_w13_scales = weight_tensor(
            t2_w13_scales_ptr,
            tier2_experts
            * cutlass.Int64(self.tier2.fc1.scale_k_groups)
            * cutlass.Int64(self.tier2.fc1.scale_size_n // 4),
        )
        t2_w2_scales = weight_tensor(
            t2_w2_scales_ptr,
            tier2_experts
            * cutlass.Int64(self.tier2.fc2.scale_k_groups)
            * cutlass.Int64(self.tier2.fc2.scale_size_n // 4),
        )
        t0_w13_global = weight_tensor(t0_w13_global_ptr, tier0_experts)
        t0_w2_global = weight_tensor(t0_w2_global_ptr, tier0_fc2)
        t1_w13_global = weight_tensor(t1_w13_global_ptr, tier1_experts)
        t1_w2_global = weight_tensor(t1_w2_global_ptr, tier1_fc2)
        t2_w13_global = weight_tensor(t2_w13_global_ptr, tier2_experts)
        t2_w2_global = weight_tensor(t2_w2_global_ptr, tier2_fc2)
        descriptor_map = weight_tensor(
            descriptor_map_ptr, cutlass.Int64(3) * total_experts
        )
        intermediate_rotations = weight_tensor(
            intermediate_rotations_ptr,
            total_experts * cutlass.Int64(3 * self.intermediate_size),
        )
        suh_rows = total_experts
        if cutlass.const_expr(self.driver.broadcast_suh):
            suh_rows = cutlass.Int64(1)
        gate_suh = weight_tensor(
            gate_suh_ptr, suh_rows * cutlass.Int64(self.hidden_size)
        )
        up_suh = weight_tensor(up_suh_ptr, suh_rows * cutlass.Int64(self.hidden_size))
        trellis_lut = weight_tensor(
            trellis_lut_ptr, Int64(_LUT_E4M3_VALUE_TABLE_ENTRIES)
        )
        trellis_lut_addr = get_ptr_as_int64(trellis_lut, Int32(0))
        rotation_input = weight_tensor(
            rotation_input_ptr,
            active_m.to(cutlass.Int64) * cutlass.Int64(self.hidden_size),
        )
        topk_weights = weight_tensor(
            topk_weights_ptr,
            active_m.to(cutlass.Int64) * cutlass.Int64(self.top_k),
        )
        self.kernel3(
            rotation_input,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            t1_w13,
            t1_w2,
            t1_w13_scales,
            t1_w2_scales,
            t1_w13_global,
            t1_w2_global,
            t2_w13,
            t2_w2,
            t2_w13_scales,
            t2_w2_scales,
            t2_w13_global,
            t2_w2_global,
            fc1,
            activated,
            fc2,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            trellis_lut_addr,
            tier0_num_experts,
            tier1_num_experts,
            tier2_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier2_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier2_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            tier2_up_experts,
            active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            cooperative=True,
            stream=stream,
        )

    @cute.kernel
    def kernel3(
        self,
        rotation_input: cute.Tensor,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_w13_scales: cute.Tensor,
        t0_w2_scales: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t0_w2_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_w13_scales: cute.Tensor,
        t1_w2_scales: cute.Tensor,
        t1_w13_global: cute.Tensor,
        t1_w2_global: cute.Tensor,
        t2_w13: cute.Tensor,
        t2_w2: cute.Tensor,
        t2_w13_scales: cute.Tensor,
        t2_w2_scales: cute.Tensor,
        t2_w13_global: cute.Tensor,
        t2_w2_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        trellis_lut_addr: Int64,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier2_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        tier2_fc2_experts: cutlass.Int32,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier2_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        tier2_up_experts: cutlass.Int32,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_x_raw)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        phase_lut_addr = trellis_lut_addr
        if cutlass.const_expr(self.driver.lut_e4m3_smem):
            self.driver._lut_smem_copy(
                trellis_lut_addr,
                smem_base + Int32(self.lut_e4m3_smem_off),
                _LUT_E4M3_SMEM_REGION_BYTES,
                tid,
            )
            cute.arch.sync_threads()
            phase_lut_addr = Int64(smem_base + Int32(self.lut_e4m3_smem_off))
        common = (
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            topk_weights,
        )
        counts = (
            tier0_num_experts,
            tier1_num_experts,
            tier2_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier2_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier2_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            tier2_up_experts,
        )
        fc1_emit = partial(
            self._emit_tier_tile3,
            True,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w13_scales,
            t0_w13_global,
            t1_w13,
            t1_w13_scales,
            t1_w13_global,
            t2_w13,
            t2_w13_scales,
            t2_w13_global,
            fc1,
            *common,
            fc1_scratch,
            workspace,
            phase_lut_addr,
            smem_base,
            tid,
            active_m,
            *counts,
        )
        fc2_emit = partial(
            self._emit_tier_tile3,
            False,
            activated,
            activated,
            t0_w2,
            t0_w2_scales,
            t0_w2_global,
            t1_w2,
            t1_w2_scales,
            t1_w2_global,
            t2_w2,
            t2_w2_scales,
            t2_w2_global,
            fc2,
            *common,
            fc2_scratch,
            workspace,
            phase_lut_addr,
            smem_base,
            tid,
            active_m * Int32(self.top_k),
            *counts,
        )
        total_experts = tier0_num_experts + tier1_num_experts + tier2_num_experts
        self.driver._moe_body(
            rotation_gate,
            rotation_up,
            rotation_input,
            t0_w13,
            t0_w2,
            fc1,
            activated,
            fc2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            t0_w13_global,
            Int32(0),
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            descriptor_map,
            trellis_lut_addr,
            trellis_lut_addr,
            total_experts,
            total_experts,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m,
            fc1_emit,
            fc2_emit,
        )


_CACHE: dict[tuple[object, ...], MixedTrellisCompileResult] = {}
_CACHE3: dict[tuple[object, ...], MixedTrellis3CompileResult] = {}
_ROUTE_PACK_WARMED: set[tuple[object, ...]] = set()


def _mixed_route_num_experts(
    expert_map: torch.Tensor,
    expected_route_num_experts: int,
) -> int:
    route_num_experts = int(expert_map.numel())
    if route_num_experts != int(expected_route_num_experts):
        raise ValueError(
            "mixed Trellis route map must match the compiled route namespace: "
            f"map={route_num_experts}, compiled={int(expected_route_num_experts)}"
        )
    return route_num_experts


def warmup_mixed_trellis_route_pack(
    launch: MixedTrellisCompileResult,
    buffers: MixedTrellisBuffers,
    *,
    expert_map: torch.Tensor,
) -> int:
    """Materialize every route-pack specialization reachable by ``launch``.

    Route packing buckets token capacity to powers of two. A profile pass at
    the maximum batch therefore does not cover smaller decode, speculative,
    or final-prefill-chunk buckets. Load those CUDA modules eagerly while the
    serving framework is still profiling persistent memory, so KV sizing sees
    their real driver footprint instead of discovering it under live traffic.
    """
    device = buffers.packed_route_indices.device
    if device.type != "cuda":
        raise RuntimeError("mixed Trellis route-pack warmup requires CUDA buffers")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("mixed Trellis route-pack warmup cannot run during capture")

    route_num_experts = _mixed_route_num_experts(
        expert_map, int(launch.topk_sum.route_num_experts)
    )
    warmed = 0
    pending_keys: list[tuple[object, ...]] = []
    with torch.cuda.device(device):
        device_index = int(torch.cuda.current_device())
        for token_count in route_pack_warmup_token_counts(launch.size_m):
            key = (
                device_index,
                str(launch.route_ids_dtype),
                int(token_count),
                int(launch.top_k),
                route_num_experts,
                int(launch.moe_block_size),
                True,
            )
            if key in _ROUTE_PACK_WARMED:
                continue
            dummy_topk_ids = torch.zeros(
                (token_count, launch.top_k),
                dtype=launch.route_ids_dtype,
                device=device,
            )
            pack_topk_routes_by_expert(
                dummy_topk_ids,
                launch.moe_block_size,
                route_num_experts,
                expert_map=expert_map,
                packed_route_indices=buffers.packed_route_indices,
                block_expert_ids=buffers.block_expert_ids,
                packed_route_count=buffers.packed_route_count,
                expert_offsets=buffers.expert_offsets,
                expert_counts=buffers.expert_counts,
            )
            pending_keys.append(key)
            warmed += 1
        torch.cuda.current_stream(device).synchronize()
        _ROUTE_PACK_WARMED.update(pending_keys)
    return warmed


def compile_mixed_trellis(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    tier0_num_experts: int,
    tier1_num_experts: int,
    top_k: int,
    max_m_blocks: int,
    sms: int,
    max_shared_mem: int,
    force_tile_config: tuple[int, int, int, int],
    tier0_bits: int = 3,
    tier1_bits: int = 4,
    trellis_codebook: str = "mcg",
    moe_block_size: int = 8,
    rotation_input_dtype: str = "bf16",
    route_ids_dtype: torch.dtype = torch.int32,
    broadcast_suh: bool = False,
    broadcast_svh: bool = False,
    route_num_experts: int | None = None,
) -> MixedTrellisCompileResult:
    if route_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("mixed Trellis route IDs must be int32 or int64")
    if int(size_m) * int(top_k) > torch.iinfo(torch.int32).max:
        raise ValueError("mixed Trellis routed-row count must fit in int32")
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = (
        int(value) for value in force_tile_config
    )
    trellis_codebook = str(trellis_codebook).lower()
    if fc1_tile_k < 128:
        raise ValueError(
            "mixed Trellis FC1 requires tile_k >= 128; narrower K tiles lose "
            "large-M cross-tier partial reductions"
        )
    total_experts = int(tier0_num_experts) + int(tier1_num_experts)
    if route_num_experts is None:
        route_num_experts = total_experts
    route_num_experts = int(route_num_experts)
    if route_num_experts <= 0:
        raise ValueError("mixed Trellis route_num_experts must be positive")
    grouped_m8_fc2 = int(moe_block_size) in (32, 64)

    def make_kernel(num_experts: int, bits: int) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=size_m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=fc1_tile_n,
            fc1_tile_k=fc1_tile_k,
            fc2_tile_n=fc2_tile_n,
            fc2_tile_k=fc2_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            fc2_moe_block_size=(8 if grouped_m8_fc2 else moe_block_size),
            fc2_schedule_route_block_factor=(2 if grouped_m8_fc2 else 1),
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=bits,
            trellis_codebook=trellis_codebook,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype=rotation_input_dtype,
            broadcast_suh=broadcast_suh,
            schedule_whole_tiles=True,
        )

    kernel = W4A16MixedTrellisKernel(
        driver=make_kernel(total_experts, tier0_bits),
        tier0=make_kernel(int(tier0_num_experts), int(tier0_bits)),
        tier1=make_kernel(int(tier1_num_experts), int(tier1_bits)),
    )
    # shared_words is the complete dynamically allocated MemRange used by the
    # cooperative kernel. CUDA permits a launch exactly at the device's
    # opt-in shared-memory limit; rejecting an additional 512 bytes here
    # unnecessarily excludes the stock mixed-K tile geometry at block-64.
    if kernel.shared_words * 4 > int(max_shared_mem):
        raise ValueError(
            "mixed Trellis shared-memory requirement exceeds the device limit: "
            f"required={kernel.shared_words * 4} "
            f"limit={int(max_shared_mem)}"
        )
    device = int(torch.cuda.current_device())
    cache_key = (
        "mixed_trellis",
        device,
        kernel.__cache_key__,
        str(route_ids_dtype),
        int(size_m),
        int(max_m_blocks),
    )
    topk_sum = compile_w4a16_topk_sum(
        m=size_m,
        topk=top_k,
        hidden_size=hidden_size,
        element_dtype="fp16",
        full_rotation=True,
        num_experts=total_experts,
        route_num_experts=route_num_experts,
        route_ids_dtype=route_ids_dtype,
        use_expert_map=True,
        broadcast_svh=broadcast_svh,
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        # Compilation is independent of an artifact's tier partition. Replace
        # plan metadata and the top-k launch on every cache lookup so one
        # artifact cannot leak its partition into another artifact.
        return replace(
            cached,
            topk_sum=topk_sum,
            tier0_num_experts=int(tier0_num_experts),
            tier1_num_experts=int(tier1_num_experts),
            sms=int(sms),
            broadcast_suh=bool(broadcast_suh),
            broadcast_svh=bool(broadcast_svh),
        )

    compile_m = _fake_m_for_specialization(size_m)
    compile_rows = compile_m * top_k
    fc1_cols = 2 * intermediate_size
    cutlass_dtype = cutlass.Float16
    rotation_dtype = _cutlass_element_dtype(rotation_input_dtype)

    def tensor(dtype, elements: int, *, align: int = 16):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (max(int(elements), 1),), assumed_align=align
        )

    def tier_args():
        return (
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        )

    scratch_elements = max(
        fc1_cols * compile_rows,
        hidden_size * compile_rows,
        4 * 256 * moe_block_size * 256,
    )
    compile_args = (
        make_ptr(rotation_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        *tier_args(),
        *tier_args(),
        tensor(cutlass_dtype, compile_rows * fc1_cols),
        tensor(cutlass_dtype, compile_rows * intermediate_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass.Int32, moe_block_size),
        tensor(cutlass.Int32, 1),
        tensor(cutlass.Int32, 1, align=4),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Int32, 4 * 256 + 2),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(tier0_num_experts),
        Int32(tier1_num_experts),
        # FC2 counts are independent artifact data; trace with the FC1 values.
        Int32(tier0_num_experts),
        Int32(tier1_num_experts),
        1,
        1,
        current_cuda_stream(),
        # Gate-count trace placeholders (keyword-only, last in the signature);
        # real counts ride each launch.
        Int32(tier0_num_experts),
        Int32(tier1_num_experts),
        # Up-count trace placeholders; real counts ride each launch too.
        Int32(tier0_num_experts),
        Int32(tier1_num_experts),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        *compile_args,
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.mixed_trellis", W4A16MixedTrellisKernel.ABI_VERSION, cache_key
        ),
        dsl_compile_options=OptLevel(2),
    )
    result = MixedTrellisCompileResult(
        compiled=compiled,
        topk_sum=topk_sum,
        trellis_lut=_trellis256_execution_lut(
            torch.device("cuda", device), trellis_codebook
        ),
        size_m=int(size_m),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        top_k=int(top_k),
        tier0_num_experts=int(tier0_num_experts),
        tier1_num_experts=int(tier1_num_experts),
        tier0_bits=int(tier0_bits),
        tier1_bits=int(tier1_bits),
        trellis_codebook=trellis_codebook,
        fc1_tile_k=fc1_tile_k,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_k=fc2_tile_k,
        fc2_tile_n=fc2_tile_n,
        moe_block_size=int(moe_block_size),
        fc2_moe_block_size=int(kernel.driver.fc2.moe_block_size),
        fc2_schedule_route_block_factor=int(
            kernel.driver.fc2.schedule_route_block_factor
        ),
        max_m_blocks=int(max_m_blocks),
        blocks_per_sm=int(kernel.blocks_per_sm),
        sms=int(sms),
        shared_memory_bytes=int(kernel.shared_words * 4),
        rotation_input_dtype=str(rotation_input_dtype),
        route_ids_dtype=route_ids_dtype,
        broadcast_suh=bool(broadcast_suh),
        broadcast_svh=bool(broadcast_svh),
    )
    _CACHE[cache_key] = result
    return result


def compile_mixed_trellis3(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    tier0_num_experts: int,
    tier1_num_experts: int,
    tier2_num_experts: int,
    top_k: int,
    max_m_blocks: int,
    sms: int,
    max_shared_mem: int,
    force_tile_config: tuple[int, int, int, int],
    tier0_bits: int = 3,
    tier1_bits: int = 4,
    tier2_bits: int = 5,
    trellis_codebook: str = "mcg",
    moe_block_size: int = 8,
    rotation_input_dtype: str = "bf16",
    route_ids_dtype: torch.dtype = torch.int32,
    broadcast_suh: bool = False,
    broadcast_svh: bool = False,
    route_num_experts: int | None = None,
) -> MixedTrellis3CompileResult:
    """Compile the dedicated three-bitrate cooperative Trellis grid."""

    if route_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("mixed Trellis route IDs must be int32 or int64")
    if int(size_m) * int(top_k) > torch.iinfo(torch.int32).max:
        raise ValueError("mixed Trellis routed-row count must fit in int32")
    trellis_codebook = str(trellis_codebook).lower()
    if trellis_codebook != "mcg":
        raise ValueError(
            "three-tier mixed Trellis supports only the MCG codebook; "
            f"got {trellis_codebook!r}"
        )
    counts = tuple(
        int(value)
        for value in (
            tier0_num_experts,
            tier1_num_experts,
            tier2_num_experts,
        )
    )
    if any(value <= 0 or value > _MAX_TIER_EXPERTS for value in counts):
        raise ValueError(
            "three-tier mixed Trellis requires each tier to contain 1..256 slots"
        )
    bits = tuple(int(value) for value in (tier0_bits, tier1_bits, tier2_bits))
    if set(bits) != {3, 4, 5}:
        raise ValueError(
            "three-tier mixed Trellis requires one K3, one K4, and one K5 tier"
        )
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = (
        int(value) for value in force_tile_config
    )
    if fc1_tile_k < 128:
        raise ValueError(
            "mixed Trellis FC1 requires tile_k >= 128; narrower K tiles lose "
            "large-M cross-tier partial reductions"
        )
    total_experts = sum(counts)
    if route_num_experts is None:
        route_num_experts = total_experts
    route_num_experts = int(route_num_experts)
    if route_num_experts <= 0:
        raise ValueError("mixed Trellis route_num_experts must be positive")
    grouped_m8_fc2 = int(moe_block_size) in (32, 64)

    def make_kernel(num_experts: int, trellis_bits: int) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=size_m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=fc1_tile_n,
            fc1_tile_k=fc1_tile_k,
            fc2_tile_n=fc2_tile_n,
            fc2_tile_k=fc2_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            fc2_moe_block_size=(8 if grouped_m8_fc2 else moe_block_size),
            fc2_schedule_route_block_factor=(2 if grouped_m8_fc2 else 1),
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=trellis_bits,
            trellis_codebook=trellis_codebook,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype=rotation_input_dtype,
            broadcast_suh=broadcast_suh,
            schedule_whole_tiles=True,
        )

    kernel = W4A16MixedTrellis3Kernel(
        driver=make_kernel(total_experts, bits[0]),
        tier0=make_kernel(counts[0], bits[0]),
        tier1=make_kernel(counts[1], bits[1]),
        tier2=make_kernel(counts[2], bits[2]),
    )
    if kernel.shared_words * 4 > int(max_shared_mem):
        raise ValueError(
            "mixed Trellis shared-memory requirement exceeds the device limit: "
            f"required={kernel.shared_words * 4} limit={int(max_shared_mem)}"
        )
    device = int(torch.cuda.current_device())
    cache_key = (
        "mixed_trellis3",
        device,
        kernel.__cache_key__,
        str(route_ids_dtype),
        int(size_m),
        int(max_m_blocks),
    )
    topk_sum = compile_w4a16_topk_sum(
        m=size_m,
        topk=top_k,
        hidden_size=hidden_size,
        element_dtype="fp16",
        full_rotation=True,
        num_experts=total_experts,
        route_num_experts=route_num_experts,
        route_ids_dtype=route_ids_dtype,
        use_expert_map=True,
        broadcast_svh=broadcast_svh,
    )
    cached = _CACHE3.get(cache_key)
    if cached is not None:
        return replace(
            cached,
            topk_sum=topk_sum,
            tier0_num_experts=counts[0],
            tier1_num_experts=counts[1],
            tier2_num_experts=counts[2],
            sms=int(sms),
            broadcast_suh=bool(broadcast_suh),
            broadcast_svh=bool(broadcast_svh),
        )

    compile_m = _fake_m_for_specialization(size_m)
    compile_rows = compile_m * top_k
    fc1_cols = 2 * intermediate_size
    cutlass_dtype = cutlass.Float16
    rotation_dtype = _cutlass_element_dtype(rotation_input_dtype)

    def tensor(dtype, elements: int, *, align: int = 16):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (max(int(elements), 1),), assumed_align=align
        )

    def tier_args():
        return (
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        )

    scratch_elements = max(
        fc1_cols * compile_rows,
        hidden_size * compile_rows,
        4 * 256 * moe_block_size * 256,
    )
    compile_args = (
        make_ptr(rotation_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        *tier_args(),
        *tier_args(),
        *tier_args(),
        tensor(cutlass_dtype, compile_rows * fc1_cols),
        tensor(cutlass_dtype, compile_rows * intermediate_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass.Int32, moe_block_size),
        tensor(cutlass.Int32, 1),
        tensor(cutlass.Int32, 1, align=4),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Int32, 4 * 256 + 2),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
        1,
        1,
        current_cuda_stream(),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        *compile_args,
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.mixed_trellis3",
            W4A16MixedTrellis3Kernel.ABI_VERSION,
            cache_key,
        ),
        dsl_compile_options=OptLevel(2),
    )
    result = MixedTrellis3CompileResult(
        compiled=compiled,
        topk_sum=topk_sum,
        trellis_lut=_trellis256_execution_lut(
            torch.device("cuda", device), trellis_codebook
        ),
        size_m=int(size_m),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        top_k=int(top_k),
        tier0_num_experts=counts[0],
        tier1_num_experts=counts[1],
        tier2_num_experts=counts[2],
        tier0_bits=bits[0],
        tier1_bits=bits[1],
        tier2_bits=bits[2],
        trellis_codebook=trellis_codebook,
        fc1_tile_k=fc1_tile_k,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_k=fc2_tile_k,
        fc2_tile_n=fc2_tile_n,
        moe_block_size=int(moe_block_size),
        fc2_moe_block_size=int(kernel.driver.fc2.moe_block_size),
        fc2_schedule_route_block_factor=int(
            kernel.driver.fc2.schedule_route_block_factor
        ),
        max_m_blocks=int(max_m_blocks),
        blocks_per_sm=int(kernel.blocks_per_sm),
        sms=int(sms),
        shared_memory_bytes=int(kernel.shared_words * 4),
        rotation_input_dtype=str(rotation_input_dtype),
        route_ids_dtype=route_ids_dtype,
        broadcast_suh=bool(broadcast_suh),
        broadcast_svh=bool(broadcast_svh),
    )
    _CACHE3[cache_key] = result
    return result


def _make_mixed_trellis_buffers(
    launch: MixedTrellisCompileResult | MixedTrellis3CompileResult,
    *,
    device: torch.device,
    sms: int,
    route_num_experts: int,
) -> MixedTrellisBuffers:
    capacity_rows = launch.size_m * launch.top_k
    route_slots = max_packed_route_slots(
        capacity_rows, launch.moe_block_size, route_num_experts
    )
    route_blocks = (route_slots + launch.moe_block_size - 1) // launch.moe_block_size
    if route_blocks > launch.max_m_blocks:
        raise ValueError(
            "mixed Trellis route buffers require "
            f"{route_blocks} blocks, but the launch was compiled for "
            f"{launch.max_m_blocks}"
        )
    fc1_cols = 2 * launch.intermediate_size
    # FC1 consumes rotation_gate before the grid-wide activation barrier. FC2
    # starts only after that barrier, so its output can safely reuse the same
    # storage without changing either phase's tensor layout.
    rotation_gate = torch.empty(
        (capacity_rows, launch.hidden_size), dtype=torch.float16, device=device
    )
    return MixedTrellisBuffers(
        rotation_gate=rotation_gate,
        rotation_up=torch.empty(
            (capacity_rows, launch.hidden_size), dtype=torch.float16, device=device
        ),
        fc1=torch.empty((capacity_rows, fc1_cols), dtype=torch.float16, device=device),
        activated=torch.empty(
            (capacity_rows, launch.intermediate_size),
            dtype=torch.float16,
            device=device,
        ),
        fc2=rotation_gate,
        output=torch.empty(
            (launch.size_m, launch.hidden_size), dtype=torch.float32, device=device
        ),
        packed_route_indices=torch.empty(route_slots, dtype=torch.int32, device=device),
        block_expert_ids=torch.empty(route_blocks, dtype=torch.int32, device=device),
        packed_route_count=torch.empty(1, dtype=torch.int32, device=device),
        expert_offsets=torch.empty(
            route_num_experts + 1, dtype=torch.int32, device=device
        ),
        expert_counts=torch.empty(route_num_experts, dtype=torch.int32, device=device),
        fc1_scratch=torch.empty(
            packed_gemm_scratch_elements(
                size_n=fc1_cols,
                route_slots=route_slots,
                moe_block_size=launch.moe_block_size,
                sms=sms,
            ),
            dtype=torch.float32,
            device=device,
        ),
        fc2_scratch=torch.empty(
            packed_gemm_scratch_elements(
                size_n=launch.hidden_size,
                route_slots=route_slots,
                moe_block_size=launch.moe_block_size,
                sms=sms,
            ),
            dtype=torch.float32,
            device=device,
        ),
        workspace=torch.zeros(
            max(sms * 4, launch.blocks_per_sm * sms) + 2,
            dtype=torch.int32,
            device=device,
        ),
    )


def make_mixed_trellis_buffers(
    launch: MixedTrellisCompileResult,
    *,
    device: torch.device,
    sms: int,
) -> MixedTrellisBuffers:
    return _make_mixed_trellis_buffers(
        launch,
        device=device,
        sms=sms,
        route_num_experts=int(launch.topk_sum.route_num_experts),
    )


def make_mixed_trellis3_buffers(
    launch: MixedTrellis3CompileResult,
    *,
    device: torch.device,
    sms: int,
) -> MixedTrellisBuffers:
    return _make_mixed_trellis_buffers(
        launch,
        device=device,
        sms=sms,
        route_num_experts=int(launch.topk_sum.route_num_experts),
    )


def build_ordered_maps(
    tier0_num_experts: int,
    tier1_num_experts: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tier0_num_experts = int(tier0_num_experts)
    tier1_num_experts = int(tier1_num_experts)
    return build_tiered_maps(
        range(tier0_num_experts),
        range(tier0_num_experts, tier0_num_experts + tier1_num_experts),
        device=device,
    )


# One tier-local expert index is encoded in the descriptor's low 8 bits.
_MAX_TIER_EXPERTS = 256


def build_tiered_maps(
    tier0_global_ids: Sequence[int],
    tier1_global_ids: Sequence[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build immutable route and descriptor maps for one mixed expert layer."""

    tier0_ids = tuple(int(expert_id) for expert_id in tier0_global_ids)
    tier1_ids = tuple(int(expert_id) for expert_id in tier1_global_ids)
    if len(tier0_ids) > 256 or len(tier1_ids) > 256:
        raise ValueError("each mixed Trellis tier supports at most 256 experts")
    total = len(tier0_ids) + len(tier1_ids)
    if sorted((*tier0_ids, *tier1_ids)) != list(range(total)):
        raise ValueError(
            f"mixed Trellis tier ids must be a disjoint partition of [0, {total})"
        )
    global_to_combined_host = [-1] * total
    for local_id, global_id in enumerate(tier0_ids):
        global_to_combined_host[global_id] = local_id
    for local_id, global_id in enumerate(tier1_ids):
        global_to_combined_host[global_id] = len(tier0_ids) + local_id
    global_to_combined = torch.tensor(
        global_to_combined_host, dtype=torch.int32, device=device
    )
    descriptor_row = torch.tensor(
        [*range(len(tier0_ids)), *((1 << 8) | i for i in range(len(tier1_ids)))],
        dtype=torch.int32,
        device=device,
    )
    # The descriptor table carries one row per projection (gate, up, down).
    # Per-expert tiering is the degenerate case where all three rows are
    # identical, reproducing single-row behaviour bit-for-bit.
    descriptor = descriptor_row.repeat(3).contiguous()
    # Preserve the per-tier gate/up counts encoded by the descriptor so binding
    # can reject a mismatched storage contract without a device synchronization.
    # Per-expert tiering has gate == up == the tier partition.
    counts = (len(tier0_ids), len(tier1_ids))
    descriptor._mt_projection_counts = (counts, counts)
    return global_to_combined, descriptor


def build_projection_tiered_maps(
    gate_tiers: Sequence[int],
    up_tiers: Sequence[int],
    down_tiers: Sequence[int],
    *,
    tier_slots: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build route and projection descriptors for a mixed-bitrate MoE layer.

    Each projection argument contains one tier ID per global expert. A global
    expert may use different tiers for gate, up, and down projections, so the
    route map remains an identity map and the descriptor rows carry all tier
    and tier-local slot information.

    Returns ``(global_to_combined, descriptor_map)``. ``descriptor_map`` is a
    contiguous ``int32[3 * sum(tier_slots)]`` tensor laid out as gate, up, and
    down rows. A populated entry encodes ``(tier << 8) | tier_local_index``;
    unused padded slots contain ``-1``.
    """

    projections = (
        ("gate", tuple(int(t) for t in gate_tiers)),
        ("up", tuple(int(t) for t in up_tiers)),
        ("down", tuple(int(t) for t in down_tiers)),
    )
    slots = tuple(int(value) for value in tier_slots)
    if len(slots) not in (2, 3):
        raise ValueError(
            "mixed Trellis tier_slots must contain exactly two or three counts"
        )
    if any(value < 0 or value > 256 for value in slots):
        raise ValueError("mixed Trellis tier slots must be in [0, 256]")
    num_experts = len(projections[0][1])
    rows: list[int] = []
    projection_counts: list[tuple[int, ...]] = []
    for name, tiers in projections:
        if len(tiers) != num_experts:
            raise ValueError(
                "mixed Trellis projection tier lists must agree on expert count: "
                f"{name} has {len(tiers)}, expected {num_experts}"
            )
        if any(t < 0 or t >= len(slots) for t in tiers):
            raise ValueError(
                f"mixed Trellis {name} tier ids must be in [0, {len(slots)})"
            )
        counters = [0] * len(slots)
        row = []
        for tier in tiers:
            local = counters[tier]
            counters[tier] += 1
            if local > 0xFF:
                raise ValueError(
                    f"mixed Trellis {name} tier {tier} exceeds 256 experts"
                )
            row.append((tier << 8) | local)
        projection_counts.append(tuple(counters))
        rows.extend(row)
    # The launch sizes the descriptor namespace as the sum of the tier slot
    # counts. A tier slot is max(gate_count, up_count), so with per-projection
    # tiering that sum can exceed the real expert count. Routing remains a
    # separate, exact-size namespace; only the descriptor rows are padded.
    required_fc1 = tuple(
        max(projection_counts[0][tier], projection_counts[1][tier])
        for tier in range(len(slots))
    )
    if any(slots[tier] < required_fc1[tier] for tier in range(len(slots))):
        raise ValueError(
            "mixed Trellis tier slots cannot address all gate/up locals: "
            f"slots={slots}, required={required_fc1}"
        )
    stride = sum(slots)
    if stride < num_experts:
        raise ValueError(
            f"mixed Trellis tier slots ({stride}) cannot address {num_experts} experts"
        )
    global_to_combined = torch.arange(num_experts, dtype=torch.int32, device=device)
    descriptor = torch.full((3 * stride,), -1, dtype=torch.int32, device=device)
    for row_index in range(3):
        base = row_index * num_experts
        descriptor[row_index * stride : row_index * stride + num_experts] = (
            torch.tensor(
                rows[base : base + num_experts], dtype=torch.int32, device=device
            )
        )
    # Preserve the gate/up counts encoded by the descriptor so binding can
    # reject a mismatched storage contract without a device synchronization.
    descriptor._mt_projection_counts = (
        projection_counts[0],
        projection_counts[1],
    )
    return global_to_combined, descriptor


def _check_descriptor_projection_counts(
    descriptor_map: torch.Tensor,
    total_experts: int,
    *,
    gate_counts: tuple[int, ...],
    up_counts: tuple[int, ...],
) -> None:
    """Fail closed when launch counts disagree with the descriptor map.

    The device dispatch bounds gate/up locals by the launch counts, so a
    descriptor entry at or beyond its projection's count is silently skipped
    and the corresponding FC1 half stays zero. The in-tree builders publish
    the counts their descriptor encodes; descriptors from other producers pay
    one host copy here, memoized on the tensor, so steady-state launches
    never synchronize.
    """

    encoded = getattr(descriptor_map, "_mt_projection_counts", None)
    if encoded is None:
        rows = descriptor_map.detach().cpu().view(3, total_experts)
        derived = []
        tier_count = len(gate_counts)
        for row in rows[:2]:
            live = row[row >= 0]
            encoded_tiers = live >> 8
            if bool((encoded_tiers >= tier_count).any()):
                raise ValueError(
                    "mixed Trellis descriptor contains a tier outside the "
                    f"launch range [0, {tier_count})"
                )
            derived.append(
                tuple(int((encoded_tiers == tier).sum()) for tier in range(tier_count))
            )
        encoded = (tuple(derived[0]), tuple(derived[1]))
        descriptor_map._mt_projection_counts = encoded
    expected_gate = tuple(int(value) for value in encoded[0])
    expected_up = tuple(int(value) for value in encoded[1])
    got_gate = tuple(int(value) for value in gate_counts)
    got_up = tuple(int(value) for value in up_counts)
    if got_gate != expected_gate or got_up != expected_up:
        raise ValueError(
            "mixed Trellis projection counts disagree with the descriptor "
            f"map: gate {got_gate} vs encoded {expected_gate}, up {got_up} "
            f"vs encoded {expected_up}"
        )


def _require_capture_safe_descriptor_metadata(
    descriptor_map: torch.Tensor,
) -> None:
    if torch.cuda.is_current_stream_capturing() and getattr(
        descriptor_map, "_mt_projection_counts", None
    ) is None:
        raise RuntimeError(
            "mixed Trellis capture-time binding requires descriptor projection "
            "counts prepared on the host"
        )


def combine_trellis_rotations(
    tier0: MixedTrellisTier,
    tier1: MixedTrellisTier,
    *additional_tiers: MixedTrellisTier,
) -> MixedTrellisRotations:
    """Materialize one tier-ordered table set once during model preparation."""
    tiers = (tier0, tier1, *additional_tiers)
    return MixedTrellisRotations(
        intermediate=torch.cat(
            tuple(tier.intermediate_rotations for tier in tiers), dim=0
        ).contiguous(),
        gate_suh=torch.cat(tuple(tier.gate_suh for tier in tiers), dim=0).contiguous(),
        up_suh=torch.cat(tuple(tier.up_suh for tier in tiers), dim=0).contiguous(),
        down_svh=torch.cat(tuple(tier.down_svh for tier in tiers), dim=0).contiguous(),
    )


def _validate_mixed_trellis_tier_storage(
    *,
    name: str,
    tier: MixedTrellisTier,
    expected_experts: int,
    bits: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    gate_experts: int | None = None,
    up_experts: int | None = None,
) -> None:
    """Fail closed before binding expert-sized storage as raw CuTe pointers."""
    expected_experts = int(expected_experts)
    bits = int(bits)
    # Per-projection membership lets a tier hold a different number of FC2
    # (down) experts than FC1 (gate/up) slots, so the FC2 count cannot be
    # assumed equal to expected_experts. Derive it from the W2 payload itself,
    # which is the tensor that actually carries the data, then require the
    # global-scale vector to agree. Deriving it from the scale vector instead
    # would report a malformed scale as a confusing W2 extent error.
    w2_expert_stride = (
        (int(intermediate_size) // 16) * (int(hidden_size) // 16) * (8 * bits)
    )
    w2_elements = int(tier.w2.numel())
    # The FC2 expert count is independent of the FC1 gate/up slot counts.
    # Its upper bound is the descriptor's 8-bit tier-local index range.
    if (
        tier.w2.dtype != torch.int32
        or w2_expert_stride <= 0
        or w2_elements % w2_expert_stride != 0
        or not 1 <= w2_elements // w2_expert_stride <= _MAX_TIER_EXPERTS
    ):
        raise ValueError(
            f"mixed Trellis {name}.w2 must be torch.int32 holding "
            f"1..{_MAX_TIER_EXPERTS} whole FC2 experts of "
            f"{w2_expert_stride} elements, got {w2_elements}"
        )
    fc2_experts = w2_elements // w2_expert_stride
    if gate_experts is None and up_experts is None:
        gate_experts = expected_experts
        up_experts = expected_experts
    elif gate_experts is None or up_experts is None:
        raise ValueError(
            f"mixed Trellis {name} requires paired gate_experts/up_experts"
        )
    gate_experts = int(gate_experts)
    up_experts = int(up_experts)
    if not (
        0 <= gate_experts <= expected_experts and 0 <= up_experts <= expected_experts
    ):
        raise ValueError(
            f"mixed Trellis {name} projection counts must both be in "
            f"[0, {expected_experts}], got gate={gate_experts}, up={up_experts}"
        )
    # Callers without projection counts use one gate and one up plane per tier
    # slot. Projection-aware callers couple the physical payload exactly to
    # gate_experts + up_experts. One dummy plane is permitted only when both
    # projection counts are zero.
    _proj_stride = (
        (int(hidden_size) // 16) * (int(intermediate_size) // 16) * (8 * bits)
    )
    _w13_planes = int(tier.w13.numel()) // _proj_stride if _proj_stride else 0
    _expected_w13_planes = max(gate_experts + up_experts, 1)
    if (
        tier.w13.dtype != torch.int32
        or tier.w13.device != device
        or not tier.w13.is_contiguous()
        or _proj_stride <= 0
        or int(tier.w13.numel()) % _proj_stride != 0
        or _w13_planes != _expected_w13_planes
        or int(tier.w13.data_ptr()) % 16 != 0
    ):
        raise ValueError(
            f"mixed Trellis {name}.w13 must be contiguous int32 on {device} "
            f"with exactly {_expected_w13_planes} projection planes, got "
            f"{int(tier.w13.numel())} elements"
        )
    expected = (
        (
            "w2",
            tier.w2,
            torch.int32,
            fc2_experts
            * (int(intermediate_size) // 16)
            * (int(hidden_size) // 16)
            * (8 * bits),
        ),
        # Native Trellis decodes its codebook tiles directly. These scale
        # pointers retain the prepared four-byte dummy ABI and are not the
        # per-expert K/32 scale grids used by packed weights.
        ("w13_scale", tier.w13_scale, torch.uint8, 4),
        ("w2_scale", tier.w2_scale, torch.uint8, 4),
        (
            "w13_global_scale",
            tier.w13_global_scale,
            torch.float32,
            expected_experts,
        ),
        (
            "w2_global_scale",
            tier.w2_global_scale,
            torch.float32,
            fc2_experts,
        ),
    )
    for field, tensor, expected_dtype, expected_elements in expected:
        if (
            tensor.dtype != expected_dtype
            or tensor.device != device
            or not tensor.is_contiguous()
            or int(tensor.numel()) != int(expected_elements)
            or int(tensor.data_ptr()) % 16 != 0
        ):
            raise ValueError(
                f"mixed Trellis {name}.{field} must be contiguous "
                f"{expected_dtype} on {device} with {int(expected_elements)} "
                "elements and at least 16-byte alignment"
            )


def _mixed_trellis_projection_counts(
    name: str,
    values: tuple[int, int] | None,
    defaults: tuple[int, int],
) -> tuple[int, int]:
    if values is None:
        return defaults
    if (
        not isinstance(values, tuple)
        or len(values) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        )
    ):
        raise TypeError(f"mixed Trellis {name} must contain two integer counts")
    return tuple(int(value) for value in values)


def _mixed_trellis_tier_pointers(tier: MixedTrellisTier) -> tuple[object, ...]:
    return (
        make_ptr(
            cutlass.Int32,
            tier.w13.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int32,
            tier.w2.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int32,
            tier.w13_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int32,
            tier.w2_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            tier.w13_global_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            tier.w2_global_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
    )


def bind_mixed_trellis(
    tier0: MixedTrellisTier,
    tier1: MixedTrellisTier,
    global_to_combined: torch.Tensor,
    descriptor_map: torch.Tensor,
    rotations: MixedTrellisRotations,
    launch: MixedTrellisCompileResult,
    *,
    gate_experts: tuple[int, int] | None = None,
    up_experts: tuple[int, int] | None = None,
) -> MixedTrellisBinding:
    """Validate immutable K3/K4 artifacts and bind their launch pointers."""

    _require_capture_safe_descriptor_metadata(descriptor_map)
    if (gate_experts is None) != (up_experts is None):
        raise ValueError(
            "mixed Trellis projection-tight storage requires paired "
            "gate_experts/up_experts"
        )
    counts = (
        int(launch.tier0_num_experts),
        int(launch.tier1_num_experts),
    )
    gate_counts = _mixed_trellis_projection_counts("gate_experts", gate_experts, counts)
    up_counts = _mixed_trellis_projection_counts("up_experts", up_experts, counts)
    tiers = (tier0, tier1)
    bits = (launch.tier0_bits, launch.tier1_bits)
    device = tier0.w13.device
    for tier_id, (tier, expected_experts, tier_bits) in enumerate(
        zip(tiers, counts, bits, strict=True)
    ):
        actual_experts = int(tier.num_experts)
        if actual_experts != expected_experts:
            raise ValueError(
                f"mixed Trellis tier{tier_id} has {actual_experts} experts, "
                f"expected the launch-plan count {expected_experts}"
            )
        tier_codebook = str(tier.trellis_codebook).lower()
        if tier_codebook != launch.trellis_codebook:
            raise ValueError(
                f"mixed Trellis tier{tier_id} uses codebook {tier_codebook!r}, "
                f"expected the launch-plan codebook {launch.trellis_codebook!r}"
            )
        _validate_mixed_trellis_tier_storage(
            name=f"tier{tier_id}",
            tier=tier,
            expected_experts=expected_experts,
            bits=tier_bits,
            hidden_size=launch.hidden_size,
            intermediate_size=launch.intermediate_size,
            device=device,
            gate_experts=gate_counts[tier_id],
            up_experts=up_counts[tier_id],
        )

    total_experts = sum(counts)
    route_num_experts = _mixed_route_num_experts(
        global_to_combined, int(launch.topk_sum.route_num_experts)
    )
    for name, mapping, expected_entries in (
        ("global_to_combined", global_to_combined, route_num_experts),
        ("descriptor_map", descriptor_map, 3 * total_experts),
    ):
        if (
            mapping.dtype != torch.int32
            or mapping.device != device
            or not mapping.is_contiguous()
            or int(mapping.numel()) != expected_entries
        ):
            raise ValueError(
                f"mixed Trellis {name} must be contiguous int32 on {device} "
                f"with {expected_entries} elements"
            )
    _check_descriptor_projection_counts(
        descriptor_map,
        total_experts,
        gate_counts=gate_counts,
        up_counts=up_counts,
    )
    for name, table, expected_elements in (
        (
            "intermediate rotations",
            rotations.intermediate,
            total_experts * 3 * launch.intermediate_size,
        ),
        (
            "gate SUH",
            rotations.gate_suh,
            (1 if launch.broadcast_suh else total_experts) * launch.hidden_size,
        ),
        (
            "up SUH",
            rotations.up_suh,
            (1 if launch.broadcast_suh else total_experts) * launch.hidden_size,
        ),
        (
            "down SVH",
            rotations.down_svh,
            (1 if launch.broadcast_svh else total_experts) * launch.hidden_size,
        ),
    ):
        if (
            table.dtype != torch.float16
            or table.device != device
            or not table.is_contiguous()
            or int(table.numel()) != expected_elements
            or int(table.data_ptr()) % 16 != 0
        ):
            raise ValueError(
                f"mixed Trellis {name} must be contiguous fp16 on {device} "
                f"with {expected_elements} elements and at least 16-byte alignment"
            )

    return MixedTrellisBinding(
        launch=launch,
        tiers=tiers,
        global_to_combined=global_to_combined,
        descriptor_map=descriptor_map,
        rotations=rotations,
        gate_counts=gate_counts,
        up_counts=up_counts,
        fc2_counts=tuple(int(tier.w2_global_scale.numel()) for tier in tiers),
        total_experts=total_experts,
        route_num_experts=route_num_experts,
        device=device,
        tier_pointers=tuple(_mixed_trellis_tier_pointers(tier) for tier in tiers),
        descriptor_map_ptr=make_ptr(
            cutlass.Int32,
            descriptor_map.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        intermediate_rotations_ptr=make_ptr(
            cutlass.Float16,
            rotations.intermediate.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        gate_suh_ptr=make_ptr(
            cutlass.Float16,
            rotations.gate_suh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        up_suh_ptr=make_ptr(
            cutlass.Float16,
            rotations.up_suh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        global_to_combined_ptr=make_ptr(
            cutlass.Int32,
            global_to_combined.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        down_svh_ptr=make_ptr(
            cutlass.Float16,
            rotations.down_svh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
    )


def run_bound_mixed_trellis(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    binding: MixedTrellisBinding,
    buffers: MixedTrellisBuffers,
) -> torch.Tensor:
    """Run a graph-safe K3/K4 launch from a validated binding."""

    launch = binding.launch
    if x.ndim != 2 or int(x.shape[1]) != int(launch.hidden_size):
        raise ValueError(
            "mixed Trellis input must have shape "
            f"[M, {int(launch.hidden_size)}], got {tuple(x.shape)}"
        )
    m = int(x.shape[0])
    if m <= 0:
        raise ValueError(f"mixed Trellis requires at least one active row, got {m}")
    if m > launch.size_m:
        raise ValueError(f"active rows {m} exceed launch capacity {launch.size_m}")
    expected_route_shape = (m, int(launch.top_k))
    for name, tensor in (("topk_ids", topk_ids), ("topk_weights", topk_weights)):
        if tuple(tensor.shape) != expected_route_shape:
            raise ValueError(
                f"mixed Trellis {name} must have shape {expected_route_shape}, "
                f"got {tuple(tensor.shape)}"
            )
    expected_input_dtype = (
        torch.bfloat16 if launch.rotation_input_dtype == "bf16" else torch.float16
    )
    for name, tensor, expected_dtype in (
        ("input", x, expected_input_dtype),
        ("topk_ids", topk_ids, launch.route_ids_dtype),
        ("topk_weights", topk_weights, torch.float32),
    ):
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"mixed Trellis {name} must be {expected_dtype}, got {tensor.dtype}"
            )
        if tensor.device != binding.device:
            raise ValueError(
                f"mixed Trellis {name} must be on {binding.device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"mixed Trellis {name} must be contiguous")
    if int(x.data_ptr()) % 16 != 0:
        raise ValueError("mixed Trellis input must have at least 16-byte alignment")

    required_route_slots = max_packed_route_slots(
        m * launch.top_k, launch.moe_block_size, binding.route_num_experts
    )
    required_route_blocks = (
        required_route_slots + launch.moe_block_size - 1
    ) // launch.moe_block_size
    if required_route_blocks > launch.max_m_blocks:
        raise RuntimeError(
            "mixed Trellis request requires "
            f"{required_route_blocks} route blocks, but the launch supports "
            f"{launch.max_m_blocks}"
        )
    if buffers.packed_route_indices.numel() < required_route_slots:
        raise RuntimeError(
            "mixed Trellis packed-route buffer is below request capacity"
        )
    if buffers.block_expert_ids.numel() < required_route_blocks:
        raise RuntimeError(
            "mixed Trellis block-expert buffer is below request capacity"
        )
    packed, block_experts, packed_count = pack_topk_routes_by_expert(
        topk_ids,
        launch.moe_block_size,
        binding.route_num_experts,
        expert_map=binding.global_to_combined,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
    )
    stream = current_cuda_stream()
    launch.compiled(
        make_ptr(
            _cutlass_element_dtype(launch.rotation_input_dtype),
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        buffers.rotation_gate.view(-1),
        buffers.rotation_up.view(-1),
        *binding.tier_pointers[0],
        *binding.tier_pointers[1],
        buffers.fc1.view(-1),
        buffers.activated.view(-1),
        buffers.fc2.view(-1),
        packed,
        block_experts,
        packed_count,
        binding.descriptor_map_ptr,
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        buffers.fc1_scratch,
        buffers.fc2_scratch,
        buffers.workspace,
        binding.intermediate_rotations_ptr,
        binding.gate_suh_ptr,
        binding.up_suh_ptr,
        make_ptr(
            cutlass.Uint8,
            launch.trellis_lut.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int32(launch.tier0_num_experts),
        Int32(launch.tier1_num_experts),
        Int32(binding.fc2_counts[0]),
        Int32(binding.fc2_counts[1]),
        m,
        max(int(launch.blocks_per_sm) * int(launch.sms), 1),
        stream,
        tier0_gate_experts=Int32(binding.gate_counts[0]),
        tier1_gate_experts=Int32(binding.gate_counts[1]),
        tier0_up_experts=Int32(binding.up_counts[0]),
        tier1_up_experts=Int32(binding.up_counts[1]),
    )
    launch.topk_sum.compiled(
        make_ptr(
            cutlass.Float16,
            buffers.fc2.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            buffers.output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Int32 if topk_ids.dtype == torch.int32 else cutlass.Int64,
            topk_ids.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4 if topk_ids.dtype == torch.int32 else 8,
        ),
        binding.global_to_combined_ptr,
        binding.down_svh_ptr,
        Int32(launch.topk_sum.num_experts),
        Int32(launch.topk_sum.route_num_experts),
        m,
        stream,
    )
    return buffers.output[:m]


def _mixed_trellis3_projection_counts(
    name: str,
    values: tuple[int, int, int] | None,
    defaults: tuple[int, int, int],
) -> tuple[int, int, int]:
    if values is None:
        return defaults
    if (
        not isinstance(values, tuple)
        or len(values) != 3
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        )
    ):
        raise TypeError(
            f"three-tier mixed Trellis {name} must contain three integer counts"
        )
    return tuple(int(value) for value in values)


def bind_mixed_trellis3(
    tier0: MixedTrellisTier,
    tier1: MixedTrellisTier,
    tier2: MixedTrellisTier,
    global_to_combined: torch.Tensor,
    descriptor_map: torch.Tensor,
    rotations: MixedTrellisRotations,
    launch: MixedTrellis3CompileResult,
    *,
    gate_experts: tuple[int, int, int] | None = None,
    up_experts: tuple[int, int, int] | None = None,
) -> MixedTrellis3Binding:
    """Validate immutable K3/K4/K5 artifacts and bind their launch pointers."""

    _require_capture_safe_descriptor_metadata(descriptor_map)
    if (gate_experts is None) != (up_experts is None):
        raise ValueError(
            "mixed Trellis projection-tight storage requires paired "
            "gate_experts/up_experts"
        )
    counts = (
        int(launch.tier0_num_experts),
        int(launch.tier1_num_experts),
        int(launch.tier2_num_experts),
    )
    gate_counts = _mixed_trellis3_projection_counts(
        "gate_experts", gate_experts, counts
    )
    up_counts = _mixed_trellis3_projection_counts("up_experts", up_experts, counts)
    tiers = (tier0, tier1, tier2)
    bits = (launch.tier0_bits, launch.tier1_bits, launch.tier2_bits)
    device = tier0.w13.device
    for tier_id, (tier, expected_experts, tier_bits) in enumerate(
        zip(tiers, counts, bits, strict=True)
    ):
        actual_experts = int(tier.num_experts)
        if actual_experts != expected_experts:
            raise ValueError(
                f"mixed Trellis tier{tier_id} has {actual_experts} experts, "
                f"expected the launch-plan count {expected_experts}"
            )
        tier_codebook = str(tier.trellis_codebook).lower()
        if tier_codebook != launch.trellis_codebook:
            raise ValueError(
                f"mixed Trellis tier{tier_id} uses codebook {tier_codebook!r}, "
                f"expected the launch-plan codebook {launch.trellis_codebook!r}"
            )
        _validate_mixed_trellis_tier_storage(
            name=f"tier{tier_id}",
            tier=tier,
            expected_experts=expected_experts,
            bits=tier_bits,
            hidden_size=launch.hidden_size,
            intermediate_size=launch.intermediate_size,
            device=device,
            gate_experts=gate_counts[tier_id],
            up_experts=up_counts[tier_id],
        )

    total_experts = sum(counts)
    route_num_experts = _mixed_route_num_experts(
        global_to_combined, int(launch.topk_sum.route_num_experts)
    )
    for name, mapping, expected_entries in (
        ("global_to_combined", global_to_combined, route_num_experts),
        ("descriptor_map", descriptor_map, 3 * total_experts),
    ):
        if (
            mapping.dtype != torch.int32
            or mapping.device != device
            or not mapping.is_contiguous()
            or int(mapping.numel()) != expected_entries
        ):
            raise ValueError(
                f"mixed Trellis {name} must be contiguous int32 on {device} "
                f"with {expected_entries} elements"
            )
    _check_descriptor_projection_counts(
        descriptor_map,
        total_experts,
        gate_counts=gate_counts,
        up_counts=up_counts,
    )
    for name, table, expected_elements in (
        (
            "intermediate rotations",
            rotations.intermediate,
            total_experts * 3 * launch.intermediate_size,
        ),
        (
            "gate SUH",
            rotations.gate_suh,
            (1 if launch.broadcast_suh else total_experts) * launch.hidden_size,
        ),
        (
            "up SUH",
            rotations.up_suh,
            (1 if launch.broadcast_suh else total_experts) * launch.hidden_size,
        ),
        (
            "down SVH",
            rotations.down_svh,
            (1 if launch.broadcast_svh else total_experts) * launch.hidden_size,
        ),
    ):
        if (
            table.dtype != torch.float16
            or table.device != device
            or not table.is_contiguous()
            or int(table.numel()) != expected_elements
            or int(table.data_ptr()) % 16 != 0
        ):
            raise ValueError(
                f"mixed Trellis {name} must be contiguous fp16 on {device} "
                f"with {expected_elements} elements and at least 16-byte alignment"
            )

    return MixedTrellis3Binding(
        launch=launch,
        tiers=tiers,
        global_to_combined=global_to_combined,
        descriptor_map=descriptor_map,
        rotations=rotations,
        gate_counts=gate_counts,
        up_counts=up_counts,
        fc2_counts=tuple(int(tier.w2_global_scale.numel()) for tier in tiers),
        total_experts=total_experts,
        route_num_experts=route_num_experts,
        device=device,
        tier_pointers=tuple(_mixed_trellis_tier_pointers(tier) for tier in tiers),
        descriptor_map_ptr=make_ptr(
            cutlass.Int32,
            descriptor_map.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        intermediate_rotations_ptr=make_ptr(
            cutlass.Float16,
            rotations.intermediate.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        gate_suh_ptr=make_ptr(
            cutlass.Float16,
            rotations.gate_suh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        up_suh_ptr=make_ptr(
            cutlass.Float16,
            rotations.up_suh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        global_to_combined_ptr=make_ptr(
            cutlass.Int32,
            global_to_combined.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        down_svh_ptr=make_ptr(
            cutlass.Float16,
            rotations.down_svh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
    )


def run_bound_mixed_trellis3(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    binding: MixedTrellis3Binding,
    buffers: MixedTrellisBuffers,
) -> torch.Tensor:
    """Run a graph-safe K3/K4/K5 launch from a validated binding."""

    launch = binding.launch
    if x.ndim != 2 or int(x.shape[1]) != int(launch.hidden_size):
        raise ValueError(
            "mixed Trellis input must have shape "
            f"[M, {int(launch.hidden_size)}], got {tuple(x.shape)}"
        )
    m = int(x.shape[0])
    if m <= 0:
        raise ValueError(f"mixed Trellis requires at least one active row, got {m}")
    if m > launch.size_m:
        raise ValueError(f"active rows {m} exceed launch capacity {launch.size_m}")
    expected_route_shape = (m, int(launch.top_k))
    for name, tensor in (("topk_ids", topk_ids), ("topk_weights", topk_weights)):
        if tuple(tensor.shape) != expected_route_shape:
            raise ValueError(
                f"mixed Trellis {name} must have shape {expected_route_shape}, "
                f"got {tuple(tensor.shape)}"
            )
    expected_input_dtype = (
        torch.bfloat16 if launch.rotation_input_dtype == "bf16" else torch.float16
    )
    for name, tensor, expected_dtype in (
        ("input", x, expected_input_dtype),
        ("topk_ids", topk_ids, launch.route_ids_dtype),
        ("topk_weights", topk_weights, torch.float32),
    ):
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"mixed Trellis {name} must be {expected_dtype}, got {tensor.dtype}"
            )
        if tensor.device != binding.device:
            raise ValueError(
                f"mixed Trellis {name} must be on {binding.device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"mixed Trellis {name} must be contiguous")
    if int(x.data_ptr()) % 16 != 0:
        raise ValueError("mixed Trellis input must have at least 16-byte alignment")

    required_route_slots = max_packed_route_slots(
        m * launch.top_k, launch.moe_block_size, binding.route_num_experts
    )
    required_route_blocks = (
        required_route_slots + launch.moe_block_size - 1
    ) // launch.moe_block_size
    if required_route_blocks > launch.max_m_blocks:
        raise RuntimeError(
            "mixed Trellis request requires "
            f"{required_route_blocks} route blocks, but the launch supports "
            f"{launch.max_m_blocks}"
        )
    if buffers.packed_route_indices.numel() < required_route_slots:
        raise RuntimeError(
            "mixed Trellis packed-route buffer is below request capacity"
        )
    if buffers.block_expert_ids.numel() < required_route_blocks:
        raise RuntimeError(
            "mixed Trellis block-expert buffer is below request capacity"
        )
    packed, block_experts, packed_count = pack_topk_routes_by_expert(
        topk_ids,
        launch.moe_block_size,
        binding.route_num_experts,
        expert_map=binding.global_to_combined,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
    )
    stream = current_cuda_stream()

    launch.compiled(
        make_ptr(
            _cutlass_element_dtype(launch.rotation_input_dtype),
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        buffers.rotation_gate.view(-1),
        buffers.rotation_up.view(-1),
        *binding.tier_pointers[0],
        *binding.tier_pointers[1],
        *binding.tier_pointers[2],
        buffers.fc1.view(-1),
        buffers.activated.view(-1),
        buffers.fc2.view(-1),
        packed,
        block_experts,
        packed_count,
        binding.descriptor_map_ptr,
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        buffers.fc1_scratch,
        buffers.fc2_scratch,
        buffers.workspace,
        binding.intermediate_rotations_ptr,
        binding.gate_suh_ptr,
        binding.up_suh_ptr,
        make_ptr(
            cutlass.Uint8,
            launch.trellis_lut.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int32(int(launch.tier0_num_experts)),
        Int32(int(launch.tier1_num_experts)),
        Int32(int(launch.tier2_num_experts)),
        Int32(binding.fc2_counts[0]),
        Int32(binding.fc2_counts[1]),
        Int32(binding.fc2_counts[2]),
        m,
        max(int(launch.blocks_per_sm) * int(launch.sms), 1),
        stream,
        tier0_gate_experts=Int32(binding.gate_counts[0]),
        tier1_gate_experts=Int32(binding.gate_counts[1]),
        tier2_gate_experts=Int32(binding.gate_counts[2]),
        tier0_up_experts=Int32(binding.up_counts[0]),
        tier1_up_experts=Int32(binding.up_counts[1]),
        tier2_up_experts=Int32(binding.up_counts[2]),
    )
    launch.topk_sum.compiled(
        make_ptr(
            cutlass.Float16,
            buffers.fc2.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            buffers.output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Int32 if topk_ids.dtype == torch.int32 else cutlass.Int64,
            topk_ids.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4 if topk_ids.dtype == torch.int32 else 8,
        ),
        binding.global_to_combined_ptr,
        binding.down_svh_ptr,
        Int32(launch.topk_sum.num_experts),
        Int32(launch.topk_sum.route_num_experts),
        m,
        stream,
    )
    return buffers.output[:m]


__all__ = [
    "MixedTrellisBuffers",
    "MixedTrellisBinding",
    "MixedTrellisCompileResult",
    "MixedTrellis3CompileResult",
    "MixedTrellis3Binding",
    "MixedTrellisRotations",
    "W4A16MixedTrellis3Kernel",
    "bind_mixed_trellis",
    "bind_mixed_trellis3",
    "build_ordered_maps",
    "build_projection_tiered_maps",
    "build_tiered_maps",
    "combine_trellis_rotations",
    "compile_mixed_trellis",
    "compile_mixed_trellis3",
    "make_mixed_trellis_buffers",
    "make_mixed_trellis3_buffers",
    "run_bound_mixed_trellis",
    "run_bound_mixed_trellis3",
    "warmup_mixed_trellis_route_pack",
]
