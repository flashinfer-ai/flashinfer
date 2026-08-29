"""Token communication implementations for MegaMoE-style kernels.

Current implementation: token-in pull with token-back push.  The standalone
``dispatch_kernel`` uses the same object methods as the fused MegaMoE kernel.
"""

import dataclasses
import os
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import (
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass.base_dsl.dsl import extract_mlir_attributes

try:
    from cutlass.cute import iket as _iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from .iket_compat import iket as _iket

from .grid_sync import software_grid_sync
from .ptx_helpers import (
    cp_reduce_async_bulk_add_noftz_bf16_s2g,
    fns_b32,
    ldg_b32_raw,
    ldg_f32_raw,
    read_clock64,
    red_add_relaxed_sys_u64_raw,
    red_add_release_gpu_s32,
    red_add_release_gpu_u64_raw,
    red_add_release_sys_s32_raw,
    red_add_release_sys_u64_raw,
    stg_b32_raw,
    stg_b64_raw,
    tma_load_1d_raw,
    tma_store_1d,
    _fence_rel_sys,
)
from .flag_batch import GpuReleaseFlagBatchTracker
from .sf_swizzle import sf_atom_int32_offset
from cutlass._mlir import ir
from common.megamoe_constants import Fp32Max, Fp8E4M3RcpLimit, Fp8E5M2RcpLimit
from common.moe_utils import cvt_f32_to_f8_to_f32
from moe_nvfp4_swapab.moe_utils import _nanosleep, spin_wait


# ---------------------------------------------------------------------------
# Low-precision combine wire format (the central driver for the token-back
# quantized combine path: fc2 epilogue encoder, token_comm push, and the
# topk_reduce receiver all describe themselves through one CombineFormat).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CombineFormat:
    """Wire format of the cross-rank combine (token-back) payload.

    The fc2 epilogue quantizes each token's hidden vector into a packed data
    plane (``combine_quant``) plus a per-block scale plane (``combine_sf``); the
    receiver (topk_reduce) dequantizes and reduces over topk. The consumer is
    ALWAYS a software dequant -- never a tensor-core MMA -- so the format is
    fully self-defined here, with no hardware scale-layout constraint.

    Canonical string ``"{scale_block}{act}x{scale}"`` (leading number = scale
    block size in hidden elements), e.g. ``"16e2m1xbf16"`` (per-16 bf16 amax +
    fp4 data) or ``"32e4m3xe8m0"`` (standard MXFP8). ``"bf16"`` is the
    no-staging baseline: bf16 fc2 terms reduced directly, no scale plane.
    """

    # Element/scale tag <-> cuTe dtype.  Only the dtypes a real format uses
    # today (the bf16 baseline is ``scale_dtype is None``); extend when a new
    # format is actually added rather than ahead of need.
    _act_by_tag: ClassVar[Dict[str, type]] = {
        "e2m1": cutlass.Float4E2M1FN,
        "e4m3": cutlass.Float8E4M3FN,
        "e5m2": cutlass.Float8E5M2,
    }
    _scale_by_tag: ClassVar[Dict[str, type]] = {
        "bf16": cutlass.BFloat16,
        "e8m0": cutlass.Float8E8M0FNU,
    }

    act_dtype: type              # cuTe dtype of the packed data-plane element
    scale_dtype: Optional[type]  # cuTe dtype of a scale entry; None == bf16 baseline
    scale_block: Optional[int]   # hidden elements per scale entry; None == baseline

    def __post_init__(self):
        allowed_act = {cutlass.BFloat16, *self._act_by_tag.values()}
        if self.act_dtype not in allowed_act:
            raise ValueError(f"combine act_dtype {self.act_dtype} not in {allowed_act}.")
        allowed_scale = {None, *self._scale_by_tag.values()}
        if self.scale_dtype not in allowed_scale:
            raise ValueError(f"combine scale_dtype {self.scale_dtype} not in {allowed_scale}.")
        if self.scale_dtype is None:                       # bf16 no-staging baseline
            if self.act_dtype is not cutlass.BFloat16 or self.scale_block is not None:
                raise ValueError("baseline must be bf16 act with scale_block=None.")
            return
        if self.act_dtype is cutlass.BFloat16:
            raise ValueError("a quantized combine cannot use a bf16 act dtype.")
        # The scale dtype pins the block: per-16 bf16 amax / per-32 e8m0 power-of-two.
        if self.scale_dtype is cutlass.BFloat16 and self.scale_block != 16:
            raise ValueError("bf16 amax scale requires scale_block == 16.")
        if self.scale_dtype is cutlass.Float8E8M0FNU and self.scale_block != 32:
            raise ValueError("e8m0 scale requires scale_block == 32.")

    @property
    def is_quantized(self) -> bool:
        """``False`` for the bf16 (no-staging) baseline."""
        return self.scale_dtype is not None

    @property
    def name(self) -> str:
        if not self.is_quantized:
            return "bf16"
        act_tag = next(t for t, d in self._act_by_tag.items() if d is self.act_dtype)
        scale_tag = next(t for t, d in self._scale_by_tag.items() if d is self.scale_dtype)
        return f"{self.scale_block}{act_tag}x{scale_tag}"

    def __str__(self) -> str:
        return self.name

    @classmethod
    def parse(cls, text: str) -> "CombineFormat":
        """Build a CombineFormat from its canonical string (the argparser entry).

        Only the handful of supported wire formats are accepted; each key is the
        exact string ``name`` produces (so ``parse(str(fmt)) == fmt``).
        """
        # (act_dtype, scale_dtype, scale_block); None scale == bf16 baseline.
        specs = {
            "bf16":        (cutlass.BFloat16,     None,                  None),
            "16e2m1xbf16": (cutlass.Float4E2M1FN, cutlass.BFloat16,      16),
            "32e4m3xe8m0": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
            "32e5m2xe8m0": (cutlass.Float8E5M2,   cutlass.Float8E8M0FNU, 32),
        }
        token = text.strip().lower()
        if token not in specs:
            raise ValueError(
                f"invalid combine_format {text!r}: expected one of {tuple(specs)}."
            )
        act_dtype, scale_dtype, scale_block = specs[token]
        return cls(act_dtype=act_dtype, scale_dtype=scale_dtype, scale_block=scale_block)


@dataclasses.dataclass(frozen=True)
class TokenSrcMetadata:
    """Per pool-token routing record: written by token-in, read by token-back
    and the fc2 combine-redirect epilogue.

    Wire format is one i64: low 32b = ``src_token`` (needs full width); high 32b
    = ``(src_rank << 16) | src_topk`` (``src_rank < world_size`` and
    ``src_topk < num_topk`` both fit in 16b).  ``load`` / ``store`` accept either
    a ``cute.Pointer`` or a raw ``Int64`` byte address.
    """

    src_rank: Int32
    src_token: Int32
    src_topk: Int32

    nbytes: ClassVar[int] = 8

    def _pack(self) -> Int64:
        hi = (Int64(self.src_rank) << Int64(16)) | Int64(self.src_topk)
        return (hi << Int64(32)) | (Int64(self.src_token) & Int64(0xFFFFFFFF))

    @staticmethod
    def _i64_ptr(addr: Union[cute.Pointer, Int64]) -> cute.Pointer:
        addr_i = addr if isinstance(addr, Int64) else addr.toint()
        return cute.make_ptr(Int64, addr_i, AddressSpace.gmem, assumed_align=8)

    def store(self, addr: Union[cute.Pointer, Int64]) -> None:
        cute.arch.store(self._i64_ptr(addr), self._pack(), scope="gpu")

    @classmethod
    def load(cls, addr: Union[cute.Pointer, Int64]) -> "TokenSrcMetadata":
        v = Int64(cute.arch.load(cls._i64_ptr(addr), Int64, scope="gpu"))
        hi = v >> Int64(32)
        return cls(
            src_rank=Int32((hi >> Int64(16)) & Int64(0xFFFF)),
            src_token=Int32(v & Int64(0xFFFFFFFF)),
            src_topk=Int32(hi & Int64(0xFFFF)),
        )


_MLIR_VALUE_FIELDS = (
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
    "carrier_row_table",
    "group_count",
    "group_rows",
    "group_done",
    "token_rank_mask",
    "dispatch_done_counter",
    "combine_output",
    "combine_sf",
    "fc2_output_workspace",
    "fc2_output_sf",
    "fc2_done_counter",
    "token_back_schedule_counter",
    "nvlink_barrier_signal",
    "nvlink_barrier_counter",
    "grid_sync_counter",
    "local_zero_prefix",
    "shared_zero_prefix",
    "peer_rank_ptr_mapper",
    "local_rank",
)

_CONST_FIELDS = (
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

class TokenCommArgs:
    """MegaMoE token communication argument bundle."""

    def __init__(
        self,
        *,
        input_token_buffer: cute.Tensor,
        input_sf_buffer: cute.Tensor,
        topk_idx: cute.Tensor,
        input_topk_weights_buffer: cute.Tensor,
        expert_send_count: cute.Tensor,
        expert_recv_count: cute.Tensor,
        expert_recv_count_sum: cute.Tensor,
        src_token_topk_idx: cute.Tensor,
        fc1_input_token_buffer: cute.Tensor,
        fc1_input_sf_buffer: cute.Tensor,
        fc1_input_topk_weights_buffer: cute.Tensor,
        fc1_ready_counter: cute.Tensor,
        token_src_metadata: cute.Tensor,
        combine_output: cute.Tensor,
        nvlink_barrier_signal: cute.Tensor,
        nvlink_barrier_counter: cute.Tensor,
        grid_sync_counter: cute.Tensor,
        local_zero_prefix: cute.Tensor,
        shared_zero_prefix: cute.Tensor,
        peer_rank_ptr_mapper: Any,
        world_size: int,
        local_rank: int,
        num_total_experts: int,
        num_experts_per_rank: int,
        num_topk: int,
        hidden_bytes: int,
        sf_uint32_per_token: int,
        token_padding_block: int,
        sf_padding_block: int,
        sm_count: int,
        fc2_output_workspace: cute.Tensor = None,
        fc2_done_counter: cute.Tensor = None,
        token_back_schedule_counter: cute.Pointer = None,
        combine_sf: cute.Tensor = None,
        fc2_output_sf: cute.Tensor = None,
        carrier_row_table: cute.Tensor = None,
        group_count: cute.Tensor = None,
        group_rows: cute.Tensor = None,
        group_done: cute.Tensor = None,
        token_rank_mask: cute.Tensor = None,
        dispatch_done_counter: cute.Tensor = None,
    ):
        self.input_token_buffer = input_token_buffer
        self.input_sf_buffer = input_sf_buffer
        self.topk_idx = topk_idx
        self.input_topk_weights_buffer = input_topk_weights_buffer
        self.expert_send_count = expert_send_count
        self.expert_recv_count = expert_recv_count
        self.expert_recv_count_sum = expert_recv_count_sum
        self.src_token_topk_idx = src_token_topk_idx
        self.fc1_input_token_buffer = fc1_input_token_buffer
        self.fc1_input_sf_buffer = fc1_input_sf_buffer
        self.fc1_input_topk_weights_buffer = fc1_input_topk_weights_buffer
        self.fc1_ready_counter = fc1_ready_counter
        self.token_src_metadata = token_src_metadata
        self.carrier_row_table = carrier_row_table
        self.group_count = group_count
        self.group_rows = group_rows
        self.group_done = group_done
        self.token_rank_mask = token_rank_mask
        self.dispatch_done_counter = dispatch_done_counter
        self.combine_output = combine_output
        self.combine_sf = combine_sf
        self.fc2_output_workspace = fc2_output_workspace
        self.fc2_output_sf = fc2_output_sf
        self.fc2_done_counter = fc2_done_counter
        self.token_back_schedule_counter = token_back_schedule_counter
        self.nvlink_barrier_signal = nvlink_barrier_signal
        self.nvlink_barrier_counter = nvlink_barrier_counter
        self.grid_sync_counter = grid_sync_counter
        self.local_zero_prefix = local_zero_prefix
        self.shared_zero_prefix = shared_zero_prefix
        self.peer_rank_ptr_mapper = peer_rank_ptr_mapper
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_total_experts = num_total_experts
        self.num_experts_per_rank = num_experts_per_rank
        self.num_topk = num_topk
        self.hidden_bytes = hidden_bytes
        self.sf_uint32_per_token = sf_uint32_per_token
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.sm_count = sm_count

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for name in _MLIR_VALUE_FIELDS:
            attr = getattr(self, name)
            if attr is None:
                continue
            values.extend(extract_mlir_values(attr))
        return values

    def __extract_mlir_attributes__(self) -> List[Any]:
        # Mirror __extract_mlir_values__ 1:1 so per-arg attrs stay aligned; the
        # only non-empty entry is peer_rank_ptr_mapper's byval/grid_constant.
        attrs: List[Any] = []
        for name in _MLIR_VALUE_FIELDS:
            attr = getattr(self, name)
            if attr is None:
                continue
            attrs.extend(extract_mlir_attributes(attr))
        return attrs

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "TokenCommArgs":
        idx = 0
        rebuilt: Dict[str, Any] = {}
        for name in _MLIR_VALUE_FIELDS:
            proto = getattr(self, name)
            if proto is None:
                rebuilt[name] = None
                continue
            n = len(extract_mlir_values(proto))
            rebuilt[name] = new_from_mlir_values(proto, values[idx : idx + n])
            idx += n
        assert idx == len(values), (
            f"TokenCommArgs serialization mismatch: "
            f"consumed={idx} provided={len(values)}"
        )
        const_kwargs = {name: getattr(self, name) for name in _CONST_FIELDS}
        return TokenCommArgs(**rebuilt, **const_kwargs)

class TokenInPullTokenBackPush:
    """Current implementation: token-in pull, token-back push."""

    num_dispatch_warps: int = 4
    warp_threads: int = 32
    num_dispatch_threads: int = num_dispatch_warps * warp_threads
    dispatch_intra_cta_bar_id: int = 10
    kernel_tail_named_barrier_id: int = 8
    dispatch_to_sched_named_barrier_id: int = 9
    # dispatch_to_sched / kernel_tail thread counts are per-instance (see __init__).
    experts_per_dispatch_pass: int = num_dispatch_threads
    # Developer-only knob (MEGA_TOKEN_BACK_ATOMIC_BATCH); not user-facing.
    token_back_atomic_batch: int = int(
        os.environ.get("MEGA_TOKEN_BACK_ATOMIC_BATCH", "1")
    )


    def __init__(
        self,
        *,
        world_size: int,
        num_topk: int,
        num_experts_per_rank: int,
        num_total_experts: int,
        hidden: int,
        fc1_token_dtype,
        sf_uint32_per_token: int,
        token_padding_block: int,
        sf_padding_block: int,
        cluster_tile_tokens: int,
        cluster_shape_mn,
        dispatch_warp_start: int,
        num_other_warps: int,
        combine_format: "CombineFormat" = None,
        token_back_by_dispatch: bool = False,
        fc2_publishes_per_token_cluster_tile: int = 0,
        token_back_reduce_topk: bool = False,
        token_back_standalone: bool = False,
        flag_batch: int = 1,
        is_swap_ab: bool = False,
        sf_atom_swizzled: bool = True,
        token_back_schedule_mode: Literal["static", "atomic_counter"] = "static",
        dedup_dispatch: bool = False,
        max_tokens_per_rank: int = 0,
        grouped_token_back: bool = False,
    ) -> None:
        self.world_size = world_size
        self.num_topk = num_topk
        self.num_experts_per_rank = num_experts_per_rank
        self.num_total_experts = num_total_experts
        self.hidden = hidden
        self.fc1_token_dtype = fc1_token_dtype
        self.hidden_bytes = hidden * int(fc1_token_dtype.width) // 8
        self.sf_uint32_per_token = sf_uint32_per_token
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.sf_atom_swizzled = sf_atom_swizzled
        self.cluster_tile_tokens = cluster_tile_tokens
        self.cluster_shape_mn = cluster_shape_mn
        if flag_batch < 1 or flag_batch > 32:
            raise ValueError(f"flag_batch must be in [1, 32], got {flag_batch}.")
        # Release-flag batch size consumed by dispatch_pull as a Python int.
        # One warp lane carries one delayed release target.
        self._flag_batch = flag_batch
        self.is_swap_ab = is_swap_ab

        # Wire-level top-k dedup: a token whose top-k hits several experts on
        # the same remote rank crosses NVLink once (the smallest-expert route
        # is the carrier); the duplicate routes copy the carrier's pool row
        # locally.  Route words gain flag bits 31 (duplicate) / 30 (carrier
        # with duplicates), so the payload index space must fit in 30 bits.
        self.dedup_dispatch = dedup_dispatch
        self.max_tokens_per_rank = max_tokens_per_rank
        if dedup_dispatch:
            if max_tokens_per_rank <= 0:
                raise ValueError(
                    "dedup_dispatch requires max_tokens_per_rank > 0 "
                    "(carrier-table extent)."
                )
            if max_tokens_per_rank * num_topk >= (1 << 30):
                raise ValueError(
                    "dedup_dispatch route words reserve bits 30/31 for flags; "
                    f"max_tokens_per_rank*num_topk={max_tokens_per_rank * num_topk} "
                    "must stay below 2**30."
                )

        # Grouped token-back (combine dedup): all pool rows of one
        # (src_rank, src_token) group are pre-reduced in fp32 by the LAST row
        # to finish fc2, and one row per contributing rank crosses the wire
        # into the source's [t_cap][world_size] inbox (quantized when the
        # combine format is).  Requires the dispatch token-back DATA path
        # (fc2 rows staged locally) and topk weights folded into FC1.
        self.grouped_token_back = grouped_token_back
        if grouped_token_back:
            if not token_back_by_dispatch:
                raise ValueError(
                    "grouped_token_back requires a dispatch token-back mode "
                    "(reuse_dispatch_warps); epi_warps pushes rows straight "
                    "from the fc2 epilogue and cannot group them."
                )
            if token_back_standalone:
                raise NotImplementedError(
                    "grouped_token_back currently runs on the dispatch warps "
                    "(reuse_dispatch_warps); the 32-register standalone "
                    "token-back warps cannot hold the fp32 group reduction."
                )
            if token_back_reduce_topk:
                raise ValueError(
                    "grouped_token_back replaces token_back_reduce_topk "
                    "(both collapse the topk axis); enable only one."
                )
            if max_tokens_per_rank <= 0:
                raise ValueError(
                    "grouped_token_back requires max_tokens_per_rank > 0 "
                    "(group-table extent)."
                )
            if hidden % 512 != 0:
                raise ValueError(
                    "grouped_token_back reduces in 512-element chunks "
                    f"(16 per lane); hidden={hidden} must be a multiple of 512."
                )

        if token_back_schedule_mode not in ("static", "atomic_counter"):
            raise ValueError(
                "token_back_schedule_mode must be 'static' or "
                f"'atomic_counter'; got {token_back_schedule_mode!r}."
            )
        self.token_back_schedule_mode = token_back_schedule_mode
        self.dispatch_warp_start = dispatch_warp_start
        # Warps that share this CTA with the dispatch group but are not part
        # of it. They participate in kernel-tail / dispatch-with-other
        # rendezvous and determine `number_of_threads` for those barriers.
        # Pure standalone dispatch passes 0 (no cohabitants -> barriers
        # collapse to dispatch-only).
        self.num_other_warps = num_other_warps
        self.num_other_threads = num_other_warps * self.warp_threads

        if combine_format is None:
            combine_format = CombineFormat(
                act_dtype=cutlass.BFloat16, scale_dtype=None, scale_block=None,
            )
        self.combine_format = combine_format
        self.token_back_by_dispatch = token_back_by_dispatch
        self.push_data = token_back_by_dispatch
        self.push_sf = combine_format.is_quantized

        # Standalone token-back: a dedicated warpgroup (size == dispatch group)
        self.token_back_standalone = token_back_standalone
        self.num_token_back_warps = self.num_dispatch_warps if self.token_back_standalone else 0
        self.num_token_back_threads = self.num_token_back_warps * self.warp_threads
        self.token_back_warp_start = dispatch_warp_start + self.num_dispatch_warps
        # Standalone token-back per-warp pull buffer; token is moved in
        # tb_chunk_bytes pieces (last piece carries the remainder), so this is
        # independent of hidden.
        self.tb_chunk_bytes = 2048

        self.num_total_threads = (
            self.num_dispatch_threads
            + self.num_other_threads
            + self.num_token_back_threads
        )
        self.dispatch_to_sched_threads = (
            self.num_dispatch_warps + 1 + self.num_token_back_warps
        ) * self.warp_threads
        self.kernel_tail_threads = self.num_total_threads

        # The DATA wire dtype is the combine act dtype (bf16 baseline -> bf16;
        # fp4/e4m3 quantized), NOT the kernel's fc2 output dtype: the cross-rank
        # payload is what the receiver dequantizes.
        self.fc2_output_dtype = combine_format.act_dtype
        if token_back_reduce_topk:
            if not token_back_by_dispatch:
                raise ValueError(
                    "token_back_reduce_topk=True requires the dispatch "
                    "token-back DATA path (token_back_by_dispatch)."
                )
            if combine_format.act_dtype is not cutlass.BFloat16:
                raise NotImplementedError(
                    "token_back_reduce_topk currently supports a bf16 combine "
                    f"only, got {combine_format}."
                )
        self.token_back_reduce_topk = token_back_reduce_topk
        if self.enable_token_back:
            self.fc2_token_bytes = hidden * int(combine_format.act_dtype.width) // 8
            if self.fc2_token_bytes % self.hidden_bytes != 0:
                raise ValueError(
                    f"fc2_token_bytes={self.fc2_token_bytes} must be a "
                    f"multiple of hidden_bytes={self.hidden_bytes} so the "
                    f"per-warp pull buffer can be reused chunk-by-chunk."
                )
            self.fc2_num_chunks = self.fc2_token_bytes // self.hidden_bytes
            if fc2_publishes_per_token_cluster_tile <= 0:
                raise ValueError(
                    "fc2_publishes_per_token_cluster_tile must be > 0 when "
                    "token-back is enabled (it gates the per-expert push)."
                )
            self.fc2_publishes_per_token_cluster_tile = (
                fc2_publishes_per_token_cluster_tile
            )
        else:
            self.fc2_token_bytes = 0
            self.fc2_num_chunks = 0
            self.fc2_publishes_per_token_cluster_tile = 0

    @property
    def enable_token_back(self) -> bool:
        # token-back warps run if they push the DATA plane, the SF plane, or both.
        return self.push_data or self.push_sf

    def extra_smem_storage_class(self) -> type:
        hidden_bytes = self.hidden_bytes
        num_total_experts = self.num_total_experts

        if self.token_back_standalone:
            @cute.struct
            class TokenCommStorage:
                pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
                smem_expert_count: cute.struct.MemRange[
                    Int32, num_total_experts
                ]
                pull_buffer: cute.struct.Align[cute.struct.MemRange[Uint8, self.num_dispatch_warps * hidden_bytes], 16]
                tb_pull_mbar: cute.struct.MemRange[Int64, self.num_token_back_warps]
                tb_pull_buffer: cute.struct.Align[cute.struct.MemRange[Uint8, self.num_token_back_warps * self.tb_chunk_bytes], 16]

            return TokenCommStorage

        @cute.struct
        class TokenCommStorage:
            pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
            smem_expert_count: cute.struct.MemRange[
                Int32, num_total_experts
            ]
            pull_buffer: cute.struct.Align[cute.struct.MemRange[Uint8, self.num_dispatch_warps * hidden_bytes], 16]

        return TokenCommStorage

    def fc1_ready_counter_ptr(self, token_comm_args):
        return token_comm_args.fc1_ready_counter.iterator

    @cute.jit
    def sched_warp_pre_init_wait(self, token_comm_args):
        nb = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb.arrive_and_wait()

    @cute.jit
    def fc1_tma_b_predispatch_spin(self, token_comm_args, work_tile_info):
        if cutlass.const_expr(self.is_swap_ab):
            token_cluster_size: cutlass.Constexpr = self.cluster_shape_mn[1]
            cluster_token_block_idx = (
                work_tile_info.tile_n_idx // cutlass.Int32(token_cluster_size)
            )
            counter_slot = (
                work_tile_info.cumulative_token_block_count
                + cluster_token_block_idx
            )
            if cutlass.const_expr(token_cluster_size == 1):
                # Preserve the existing NVFP4/MXFP8 single-token-CTA path.
                peek_threshold = work_tile_info.valid_tokens_in_cta_tile
            else:
                packed_expert_count = token_comm_args.expert_recv_count_sum[
                    work_tile_info.expert_idx
                ]
                expert_token_count = Int32(
                    Int64(packed_expert_count) & Int64(0xFFFFFFFF)
                )
                remaining_cluster_tokens = cutlass.max(
                    expert_token_count
                    - cluster_token_block_idx
                    * cutlass.Int32(self.cluster_tile_tokens),
                    Int32(0),
                )
                peek_threshold = cutlass.min(
                    remaining_cluster_tokens,
                    cutlass.Int32(self.cluster_tile_tokens),
                )
        else:
            counter_slot = (
                work_tile_info.cumulative_token_block_count
                + work_tile_info.tile_m_idx // cutlass.Int32(self.cluster_shape_mn[0])
            )
            peek_threshold = work_tile_info.valid_tokens_in_cluster_tile

        counter_ptr = token_comm_args.fc1_ready_counter.iterator + counter_slot
        # Dispatch warps may fill rows within a cluster token block out of
        # order. Every valid CTA therefore waits for the full cluster-block
        # count; an invalid tail CTA has no rows to consume and skips the wait.
        if work_tile_info.valid_tokens_in_cta_tile > Int32(0):
            if not work_tile_info.peek_ready:
                _iket.range_push("tma_token_fc1_wait")
                spin_wait(
                    counter_ptr,
                    lambda v: v >= peek_threshold,
                    fail_sleep_cycles=1000,
                )
                _iket.range_pop()

    @cute.jit
    def dispatch_prep(
        self,
        token_comm_storage,
        topk_idx,
        expert_send_count,
        src_token_topk_idx,
        token_rank_mask,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_tokens,
        num_sms,
    ):
        thread_idx_in_dispatch = Int32(warp_idx * self.warp_threads + lane_idx)
        smem_count_ptr = token_comm_storage.smem_expert_count.data_ptr()
        i = thread_idx_in_dispatch
        while i < Int32(self.num_total_experts):
            (smem_count_ptr + i).store(Int32(0))
            i = i + Int32(self.num_dispatch_threads)
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.num_dispatch_threads,
        )

        tokens_per_warp: cutlass.Constexpr[int] = 32 // self.num_topk
        active_lanes: cutlass.Constexpr[int] = tokens_per_warp * self.num_topk
        num_dispatch_warps_per_grid: cutlass.Constexpr[int] = num_sms * self.num_dispatch_warps

        base_token_for_warp = (sm_idx * self.num_dispatch_warps + warp_idx) * tokens_per_warp
        grid_token_stride = num_dispatch_warps_per_grid * tokens_per_warp

        t = base_token_for_warp
        while t < num_tokens:
            token_offset_in_warp = lane_idx // self.num_topk
            token_global = t + token_offset_in_warp
            if lane_idx < active_lanes and token_global < num_tokens:
                topk_slot = lane_idx % self.num_topk
                expert_id = Int32(topk_idx[token_global, topk_slot])
                if expert_id >= Int32(0):
                    cute.arch.atomic_add(
                        smem_count_ptr + expert_id,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
            cute.arch.sync_warp()
            t += grid_token_stride

        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.num_dispatch_threads,
        )

        for offset in cutlass.range_constexpr(
            0, self.num_total_experts, self.experts_per_dispatch_pass,
        ):
            expert_id = Int32(offset + warp_idx * self.warp_threads + lane_idx)
            if expert_id < Int32(self.num_total_experts):
                slot_ptr = smem_count_ptr + expert_id
                local_count = (slot_ptr).load()
                delta = (Int64(1) << Int64(32)) | (Int64(local_count) & Int64(0xFFFFFFFF))
                old_packed = cute.arch.atomic_add(
                    expert_send_count.iterator + expert_id,
                    delta,
                    sem="relaxed",
                    scope="gpu",
                )
                base_slot = Int32(old_packed & Int64(0xFFFFFFFF))
                (slot_ptr).store(base_slot)
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.num_dispatch_threads,
        )

        if cutlass.const_expr(self.dedup_dispatch):
            t = base_token_for_warp
            while t < num_tokens:
                token_offset_in_warp = lane_idx // self.num_topk
                token_global = t + token_offset_in_warp
                topk_slot = lane_idx % self.num_topk
                expert_id = Int32(-1)
                if lane_idx < active_lanes and token_global < num_tokens:
                    expert_id = Int32(topk_idx[token_global, topk_slot])
                # Carrier election among this token's top-k lanes (they sit in
                # adjacent lanes of this warp): per destination rank, the
                # smallest expert id carries the payload; every other route to
                # that rank is a wire-duplicate the receiver resolves from the
                # carrier's pool row.  Self-rank routes are exempt (their pull
                # is already a local HBM read).  Carrier-in-a-smaller-expert is
                # what makes the receiver-side wait graph a DAG: dispatch
                # warps walk experts in ascending order, so a duplicate only
                # ever waits on a row of a strictly smaller expert.
                dst_rank = expert_id // Int32(self.num_experts_per_rank)
                base_lane = lane_idx - topk_slot
                is_dup = Int32(0)
                has_larger = Int32(0)
                suppressed = Int32(0)
                for j in cutlass.range_constexpr(0, self.num_topk, 1):
                    other = Int32(
                        cute.arch.shuffle_sync(expert_id, base_lane + Int32(j))
                    )
                    if expert_id >= Int32(0) and other >= Int32(0):
                        if dst_rank != Int32(local_rank):
                            if other // Int32(self.num_experts_per_rank) == dst_rank:
                                if other < expert_id:
                                    # Duplicates only ever wait on a strictly
                                    # smaller expert (the receiver-side wait
                                    # DAG depends on it).
                                    is_dup = Int32(1)
                                if other > expert_id:
                                    has_larger = Int32(1)
                                if (other == expert_id) and (
                                    Int32(j) < topk_slot
                                ):
                                    # Repeated expert id (not produced by
                                    # standard top-k, but never validated
                                    # against): the non-minimum-slot copies
                                    # drop out of dedup entirely -- they pull
                                    # normally and never publish, so exactly
                                    # one carrier remains (a double red.add
                                    # publish would corrupt the table entry's
                                    # valid bit and hang the waiters) and no
                                    # same-expert wait edge is created.
                                    suppressed = Int32(1)
                if expert_id >= Int32(0):
                    local_expert = expert_id % Int32(self.num_experts_per_rank)
                    slot = cute.arch.atomic_add(
                        smem_count_ptr + expert_id,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    token_topk_word = Int32(token_global * self.num_topk + topk_slot)
                    if (is_dup == Int32(1)) and (suppressed == Int32(0)):
                        # Bit 31 (sign bit as a raw flag): duplicate route.
                        token_topk_word = token_topk_word | Int32(-2147483648)
                    elif (has_larger == Int32(1)) and (
                        is_dup == Int32(0)
                    ) and (suppressed == Int32(0)):
                        # Bit 30: carrier that actually has waiting duplicates
                        # (singletons skip the table publish).
                        token_topk_word = token_topk_word | Int32(0x40000000)
                    MAX_SLOT_C: cutlass.Constexpr[int] = num_tokens * self.num_topk
                    elem_off = (
                        (local_expert * Int32(self.world_size) + Int32(local_rank))
                        * Int32(MAX_SLOT_C)
                        + slot
                    ) * Int32(4)
                    peer_addr = peer_rank_ptr_mapper.map(
                        src_token_topk_idx.iterator.toint(),
                        dst_rank, Int64(elem_off),
                    )
                    stg_b32_raw(peer_addr, token_topk_word)
                cute.arch.sync_warp()
                t += grid_token_stride
        else:
            t = base_token_for_warp
            while t < num_tokens:
                token_offset_in_warp = lane_idx // self.num_topk
                token_global = t + token_offset_in_warp
                if lane_idx < active_lanes and token_global < num_tokens:
                    topk_slot = lane_idx % self.num_topk
                    expert_id = Int32(topk_idx[token_global, topk_slot])
                    if expert_id >= Int32(0):
                        dst_rank = expert_id // Int32(self.num_experts_per_rank)
                        local_expert = expert_id % Int32(self.num_experts_per_rank)
                        slot = cute.arch.atomic_add(
                            smem_count_ptr + expert_id,
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                        token_topk_word = Int32(token_global * self.num_topk + topk_slot)
                        MAX_SLOT_C: cutlass.Constexpr[int] = num_tokens * self.num_topk
                        elem_off = (
                            (local_expert * Int32(self.world_size) + Int32(local_rank))
                            * Int32(MAX_SLOT_C)
                            + slot
                        ) * Int32(4)
                        peer_addr = peer_rank_ptr_mapper.map(
                            src_token_topk_idx.iterator.toint(),
                            dst_rank, Int64(elem_off),
                        )
                        stg_b32_raw(peer_addr, token_topk_word)
                cute.arch.sync_warp()
                t += grid_token_stride

        if cutlass.const_expr(self.grouped_token_back):
            # Contributing-rank bitmask per LOCAL token, consumed by the
            # source-side rank-slot reduce: bit r set <=> some top-k route of
            # this token lands on rank r (so exactly the inbox slots that
            # will be written).  Same lane layout as pass 2; the per-token
            # lane group ORs its rank bits via shuffles and slot-0 publishes.
            t = base_token_for_warp
            while t < num_tokens:
                token_offset_in_warp = lane_idx // self.num_topk
                token_global = t + token_offset_in_warp
                topk_slot = lane_idx % self.num_topk
                rank_bit = Int32(0)
                if lane_idx < active_lanes and token_global < num_tokens:
                    expert_id = Int32(topk_idx[token_global, topk_slot])
                    if expert_id >= Int32(0):
                        rank_bit = Int32(1) << (
                            expert_id // Int32(self.num_experts_per_rank)
                        )
                base_lane = lane_idx - topk_slot
                mask_bits = rank_bit
                for j in cutlass.range_constexpr(0, self.num_topk, 1):
                    mask_bits = mask_bits | Int32(
                        cute.arch.shuffle_sync(rank_bit, base_lane + Int32(j))
                    )
                if lane_idx < active_lanes and token_global < num_tokens:
                    if topk_slot == Int32(0):
                        token_rank_mask[token_global] = mask_bits
                cute.arch.sync_warp()
                t += grid_token_stride

    @cute.jit
    def dispatch_barrier(
        self,
        expert_send_count,
        expert_recv_count,
        expert_recv_count_sum,
        nvlink_barrier_signal,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_sms,
        nvlink_barrier_counter,
    ):
        # software_grid_sync expects a dispatch-group-relative thread id.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

        software_grid_sync(grid_sync_counter, sm_idx, num_sms, tid_in_group,
                           num_threads=self.num_dispatch_threads)

        if sm_idx == 0:
            for offset in cutlass.range_constexpr(
                0, self.num_total_experts, self.experts_per_dispatch_pass,
            ):
                expert_id = Int32(offset + warp_idx * self.warp_threads + lane_idx)
                if expert_id < Int32(self.num_total_experts):
                    dst_rank = expert_id // Int32(self.num_experts_per_rank)
                    dst_local_expert = expert_id % Int32(self.num_experts_per_rank)
                    status_u64 = cute.arch.load(
                        expert_send_count.iterator + expert_id,
                        Int64,
                        sem="relaxed",
                        scope="gpu",
                    )
                    token_count_u32 = Int32(status_u64 & Int64(0xFFFFFFFF))
                    erc_local_base = expert_recv_count.iterator.toint()
                    erc_elem_off = (
                        Int32(local_rank) * Int32(self.num_experts_per_rank) + dst_local_expert
                    ) * Int32(8)
                    erc_peer_addr = peer_rank_ptr_mapper.map(
                        erc_local_base, dst_rank, Int64(erc_elem_off),
                    )
                    stg_b64_raw(erc_peer_addr, Int64(token_count_u32))
                    ercs_local_base = expert_recv_count_sum.iterator.toint()
                    ercs_peer_addr = peer_rank_ptr_mapper.map(
                        ercs_local_base, dst_rank,
                        Int64(dst_local_expert * Int32(8)),
                    )
                    red_add_relaxed_sys_u64_raw(ercs_peer_addr, status_u64)
            cute.arch.fence_acq_rel_sys()
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.num_dispatch_threads,
        )

        self.nvlink_barrier(
            nvlink_barrier_signal,
            nvlink_barrier_counter,
            grid_sync_counter,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            num_sms=num_sms,
            prologue_grid_sync=False,
            epilogue_grid_sync=True,
        )
    @cute.jit
    def dispatch_pull(
        self,
        token_comm_storage,
        input_token_buffer,
        input_sf_buffer,
        input_topk_weights_buffer,
        src_token_topk_idx,
        expert_recv_count,
        expert_recv_count_sum,
        fc1_input_token_buffer,
        fc1_input_sf_buffer,
        fc1_input_topk_weights_buffer,
        fc1_ready_counter,
        token_src_metadata,
        carrier_row_table,
        group_count,
        group_rows,
        dispatch_done_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
    ):
        # MemRange does not support dynamic indexing here; use raw pointers.
        pull_mbar_ptr = token_comm_storage.pull_mbar.data_ptr()
        pull_buffer_ptr = token_comm_storage.pull_buffer.data_ptr()
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(pull_mbar_ptr + warp_idx, 1)
        cute.arch.sync_warp()


        phase_bit = Int32(0)

        current_expert_idx = Int32(-1)
        expert_start_idx = Int32(0)
        expert_end_idx = Int32(0)
        expert_pool_block_offset = Int32(0)
        expert_task_tile_offset = Int32(0)
        # SF rows use their own padding; token and SF pool offsets can diverge.
        expert_sf_pool_block_offset = Int32(0)

        # ── Release-flag batching ────────────────────────────────────────
        # Delay fc1-ready counter publication with the same rotating-lane
        # tracker used by the epilogue.  Each token's TMA store to the FC1 pool
        # is drained CTA-locally by ``cp_async_bulk_wait_group(0)`` before its
        # release target is accumulated; the eventual red.release.gpu add
        # publishes the corresponding pool data to GPU scope.
        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=Int32(0),
            phase=Int32(0),
            tid=lane_idx,
        )

        stored_rank_count_lane = Int32(0)

        NUM_EXPERTS_PER_LANE: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        stored_num_tokens_per_expert = []
        for _ in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            stored_num_tokens_per_expert.append(Int32(0))
        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            e_idx_for_lane = Int32(i * self.warp_threads) + lane_idx
            if e_idx_for_lane < Int32(self.num_experts_per_rank):
                sum_packed_init = expert_recv_count_sum[e_idx_for_lane]
                stored_num_tokens_per_expert[i] = Int32(
                    Int64(sum_packed_init) & Int64(0xFFFFFFFF)
                )
        cute.arch.sync_warp()

        num_global_warps: cutlass.Constexpr[int] = num_sms * self.num_dispatch_warps
        token_idx = sm_idx * Int32(self.num_dispatch_warps) + warp_idx

        _iket_pull_emit = (
            (sm_idx == Int32(0))
            and (warp_idx == Int32(0))
            and (lane_idx == Int32(0))
        )

        while current_expert_idx < Int32(self.num_experts_per_rank):
            if _iket_pull_emit:
                _iket.range_push("Pull.ChooseToken")
            old_expert_idx = current_expert_idx
            while (token_idx >= expert_end_idx) and (
                current_expert_idx < Int32(self.num_experts_per_rank)
            ):
                prev_valid_count = expert_end_idx - expert_start_idx
                prev_block_count = (
                    prev_valid_count + Int32(self.token_padding_block) - Int32(1)
                ) // Int32(self.token_padding_block)
                expert_pool_block_offset = (
                    expert_pool_block_offset + prev_block_count
                )
                # Mirror cumul for the release-counter granularity (self.cluster_tile_tokens).
                prev_task_tile_count = (
                    prev_valid_count + Int32(self.cluster_tile_tokens) - Int32(1)
                ) // Int32(self.cluster_tile_tokens)
                expert_task_tile_offset = (
                    expert_task_tile_offset + prev_task_tile_count
                )
                # Mirror cumul for the SF axis granularity (self.sf_padding_block).
                prev_sf_block_count = (
                    prev_valid_count + Int32(self.sf_padding_block) - Int32(1)
                ) // Int32(self.sf_padding_block)
                expert_sf_pool_block_offset = (
                    expert_sf_pool_block_offset + prev_sf_block_count
                )
                current_expert_idx = current_expert_idx + Int32(1)
                if current_expert_idx < Int32(self.num_experts_per_rank):
                    expert_start_idx = expert_end_idx
                    valid_value = Int32(0)
                    for i in cutlass.range_constexpr(
                        0, NUM_EXPERTS_PER_LANE, 1
                    ):
                        if current_expert_idx == Int32(i * self.warp_threads) + lane_idx:
                            valid_value = stored_num_tokens_per_expert[i]
                    total_for_expert = cute.arch.shuffle_sync(
                        valid_value, current_expert_idx % Int32(self.warp_threads)
                    )
                    expert_end_idx = expert_end_idx + total_for_expert

            if current_expert_idx < Int32(self.num_experts_per_rank):
                if old_expert_idx != current_expert_idx:
                    if lane_idx < Int32(self.world_size):
                        stored_rank_count_lane = Int32(
                            expert_recv_count[lane_idx, current_expert_idx]
                        )
                    else:
                        stored_rank_count_lane = Int32(0)

                token_idx_in_expert = token_idx - expert_start_idx
                slot_idx = token_idx_in_expert
                offset = Int32(0)
                remaining_lane = stored_rank_count_lane

                current_rank_in_expert_idx = Int32(0)
                token_idx_in_rank = Int32(0)

                decided = Int32(0)
                for _round in cutlass.range_constexpr(0, self.world_size + 1, 1):
                    if decided == Int32(0):
                        active = remaining_lane > Int32(0)
                        mask = cute.arch.vote_ballot_sync(active)
                        num_active_ranks = Int32(cute.arch.popc(Int32(mask)))
                        v_for_min = Int32(0x7FFFFFFF)
                        if active:
                            v_for_min = remaining_lane
                        length = Int32(
                            cute.arch.warp_redux_sync(v_for_min, "min")
                        )

                        if num_active_ranks > Int32(0):
                            num_round_tokens = length * num_active_ranks
                            if slot_idx < num_round_tokens:
                                slot_idx_in_round = slot_idx % num_active_ranks
                                current_rank_in_expert_idx = fns_b32(
                                    Int32(mask),
                                    Int32(0),
                                    slot_idx_in_round + Int32(1),
                                )
                                token_idx_in_rank = offset + (
                                    slot_idx // num_active_ranks
                                )
                                decided = Int32(1)
                            else:
                                slot_idx = slot_idx - num_round_tokens
                                offset = offset + length
                                if remaining_lane > length:
                                    remaining_lane = remaining_lane - length
                                else:
                                    remaining_lane = Int32(0)
                        else:
                            decided = Int32(1)

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.ChooseToken
                    _iket.range_push("Pull.TMA_NVLink_Roundtrip")

                src_token_topk = Uint32(
                    src_token_topk_idx[
                        current_expert_idx,
                        current_rank_in_expert_idx,
                        token_idx_in_rank,
                    ]
                )
                is_dup_route = Int32(0)
                publish_carrier = Int32(0)
                carrier_entry = Int64(0)
                tbl_addr_i64 = Int64(0)
                if cutlass.const_expr(self.dedup_dispatch):
                    is_dup_route = Int32(
                        (src_token_topk >> Uint32(31)) & Uint32(1)
                    )
                    publish_carrier = Int32(
                        (src_token_topk >> Uint32(30)) & Uint32(1)
                    )
                    src_token_topk = src_token_topk & Uint32(0x3FFFFFFF)
                src_token = Int32(src_token_topk // Uint32(self.num_topk))
                src_topk = Int32(src_token_topk % Uint32(self.num_topk))

                if cutlass.const_expr(self.dedup_dispatch):
                    # Rendezvous with the carrier's pool row.  Single writer
                    # per entry per launch (0 -> bit63|rows), so every lane
                    # exits with the published value; the acquire load pairs
                    # with the carrier's red.release publish, making the
                    # carrier's pool data (payload + SF) visible.  Non-dup
                    # routes never touch the table: the sentinel keeps them
                    # out of the loop, so their NVLink pull is not serialized
                    # behind an L2 round trip (the address only picks up an
                    # acquire dependency on duplicate routes, where it is
                    # required).  Loop-level loads keep every reassignment at
                    # loop scope for the DSL's while-op.
                    tbl_ptr = carrier_row_table.iterator + (
                        current_rank_in_expert_idx
                        * Int32(self.max_tokens_per_rank)
                        + src_token
                    )
                    tbl_addr_i64 = tbl_ptr.toint()
                    carrier_entry = Int64(-1)
                    if is_dup_route == Int32(1):
                        carrier_entry = Int64(0)
                    while carrier_entry >= Int64(0):
                        carrier_entry = Int64(
                            cute.arch.load(
                                tbl_ptr, Int64, sem="acquire", scope="gpu"
                            )
                        )
                        if carrier_entry >= Int64(0):
                            _nanosleep(200)

                cur_peer_offset = peer_rank_ptr_mapper.map(
                    Int64(0), current_rank_in_expert_idx, Int64(0)
                )
                inp_tok_local_base = input_token_buffer.iterator.toint()
                inp_sf_local_base = input_sf_buffer.iterator.toint()
                inp_w_local_base = input_topk_weights_buffer.iterator.toint()
                pool_sf_local_base = Int64(0)
                if cutlass.const_expr(self.dedup_dispatch):
                    pool_sf_local_base = fc1_input_sf_buffer.iterator.toint()

                data_src_addr = Int64(0)
                dedup_sf_axis = Int32(0)
                if cutlass.const_expr(self.dedup_dispatch):
                    # Duplicate route: bulk-copy the carrier's local pool row
                    # instead of re-pulling over NVLink.  Branchless select
                    # keeps every value at loop-body scope (region-nested
                    # reassignment breaks the DSL's while-op dominance).
                    data_src_addr = (
                        inp_tok_local_base
                        + cur_peer_offset
                        + Int64(src_token * Int32(self.hidden_bytes))
                    )
                    carrier_pool_row = Int32(carrier_entry & Int64(0x7FFFFFFF))
                    dedup_sf_axis = Int32(
                        (carrier_entry >> Int64(31)) & Int64(0x7FFFFFFF)
                    )
                    dedup_src_addr = (
                        fc1_input_token_buffer.iterator.toint()
                        + Int64(carrier_pool_row) * Int64(self.hidden_bytes)
                    )
                    data_src_addr = data_src_addr + (
                        dedup_src_addr - data_src_addr
                    ) * Int64(is_dup_route)

                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes)
                    )
                    if cutlass.const_expr(self.dedup_dispatch):
                        tma_load_1d_raw(
                            pull_buffer_warp_ptr,
                            data_src_addr,
                            pull_mbar_ptr + warp_idx,
                            Int32(self.hidden_bytes),
                        )
                    else:
                        tma_src_addr = (
                            inp_tok_local_base
                            + cur_peer_offset
                            + Int64(src_token * Int32(self.hidden_bytes))
                        )
                        tma_load_1d_raw(
                            pull_buffer_warp_ptr,
                            tma_src_addr,
                            pull_mbar_ptr + warp_idx,
                            Int32(self.hidden_bytes),
                        )
                cute.arch.sync_warp()

                if _iket_pull_emit:
                    _iket.range_push("Pull.SF_LDG_STG")

                sf_token_in_pool_axis = (
                    expert_sf_pool_block_offset * Int32(self.sf_padding_block)
                    + token_idx_in_expert
                )
                pool_token_idx = (
                    expert_pool_block_offset * Int32(self.token_padding_block)
                    + token_idx_in_expert
                )
                sf_passes: cutlass.Constexpr[int] = (
                    self.sf_uint32_per_token + 31
                ) // 32

                sf_vals = []
                for _ in cutlass.range_constexpr(0, sf_passes, 1):
                    sf_vals.append(Int32(0))

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        sf_addr = (
                            inp_sf_local_base
                            + cur_peer_offset
                            + Int64(
                                (src_token * Int32(self.sf_uint32_per_token) + j)
                                * Int32(4)
                            )
                        )
                        if cutlass.const_expr(self.dedup_dispatch):
                            # Duplicate route: the carrier already staged this
                            # token's SF words into the local pool (identical
                            # per-token layout for both rows); branchless
                            # address select as above.
                            if cutlass.const_expr(self.sf_atom_swizzled):
                                dedup_sf_pos = sf_atom_int32_offset(
                                    dedup_sf_axis,
                                    j,
                                    num_k_atoms=self.sf_uint32_per_token,
                                )
                            else:
                                dedup_sf_pos = (
                                    dedup_sf_axis
                                    * Int32(self.sf_uint32_per_token)
                                    + j
                                )
                            dedup_sf_addr = (
                                pool_sf_local_base
                                + Int64(dedup_sf_pos) * Int64(4)
                            )
                            sf_addr = sf_addr + (
                                dedup_sf_addr - sf_addr
                            ) * Int64(is_dup_route)
                        sf_vals[i] = ldg_b32_raw(sf_addr)

                weight = Float32(0.0)
                if lane_idx == Int32(0):
                    weight_addr = (
                        inp_w_local_base
                        + cur_peer_offset
                        + Int64(
                            (src_token * Int32(self.num_topk) + src_topk) * Int32(4)
                        )
                    )
                    weight = ldg_f32_raw(weight_addr)

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.SF_LDG_STG  (= LD phase)
                    _iket.range_push("Pull.Weight_LDG")   # (= ST phase)

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        if cutlass.const_expr(self.sf_atom_swizzled):
                            sf_int32_pos = sf_atom_int32_offset(
                                sf_token_in_pool_axis,
                                j,
                                num_k_atoms=self.sf_uint32_per_token,
                            )
                        else:
                            sf_int32_pos = (
                                sf_token_in_pool_axis
                                * Int32(self.sf_uint32_per_token)
                                + j
                            )
                        fc1_input_sf_buffer[sf_int32_pos] = sf_vals[i]
                cute.arch.sync_warp()

                if lane_idx == Int32(0):
                    fc1_input_topk_weights_buffer[pool_token_idx] = weight

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        pull_mbar_ptr + warp_idx, Int32(self.hidden_bytes)
                    )
                    cute.arch.mbarrier_wait(
                        pull_mbar_ptr + warp_idx,
                        phase_bit,
                    )

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.Weight_LDG (ST phase)
                    _iket.range_pop()  # Pull.TMA_NVLink_Roundtrip (outer)
                    _iket.range_push("Pull.TMA_Store")

                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes)
                    )
                    tma_store_1d(
                        fc1_input_token_buffer.iterator
                        # T=128k) × self.hidden_bytes overflows int32 (max 2.1 G).
                        # 64-bit address math is required for large token pools.
                        + (Int64(pool_token_idx) * Int64(self.hidden_bytes)),
                        pull_buffer_warp_ptr,
                        Int32(self.hidden_bytes),
                    )

                with cute.arch.elect_one():
                    TokenSrcMetadata(
                        src_rank=current_rank_in_expert_idx,
                        src_token=src_token,
                        src_topk=src_topk,
                    ).store(
                        token_src_metadata.iterator
                        + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                    )

                if cutlass.const_expr(self.grouped_token_back):
                    # Group table: append this pool row to its
                    # (src_rank, src_token) group.  Plain stores; visibility
                    # to the token-back readers is provided by the
                    # dispatch_done release/acquire pair below.
                    grp_gid = (
                        current_rank_in_expert_idx
                        * Int32(self.max_tokens_per_rank)
                        + src_token
                    )
                    if lane_idx == Int32(0):
                        grp_pos = cute.arch.atomic_add(
                            group_count.iterator + grp_gid,
                            Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                        group_rows[
                            grp_gid * Int32(self.num_topk) + grp_pos
                        ] = pool_token_idx

                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)

                if cutlass.const_expr(self.dedup_dispatch):
                    # Publish this row for the waiting duplicate routes.  The
                    # row's payload TMA store is drained (wait_group above,
                    # by the elected lane -- the sync_warp orders that drain
                    # before lane 0's release) and the SF/weight stores
                    # precede this point, so the red.release.gpu publish
                    # makes them visible to the duplicates' same-GPU acquire
                    # loads (same contract the fc1_ready release-flag path
                    # relies on; the table is rank-local, so device scope
                    # suffices and avoids a per-row NVLink drain).
                    cute.arch.sync_warp()
                    packed_entry = (
                        (Int64(1) << Int64(63))
                        | (Int64(sf_token_in_pool_axis) << Int64(31))
                        | Int64(pool_token_idx)
                    )
                    if (publish_carrier == Int32(1)) and (lane_idx == Int32(0)):
                        red_add_release_gpu_u64_raw(tbl_addr_i64, packed_entry)

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.TMA_Store
                    _iket.range_push("Pull.Arrival_Atomic")

                # Accumulate this token's release target into the rotating-lane
                # batch tracker.  task_tile_idx is warp-uniform (token_idx /
                # expert offsets are warp-wide), so every lane runs the same
                # state-machine transition while only one lane records the
                # current address.
                task_tile_idx = expert_task_tile_offset + (
                    token_idx_in_expert // Int32(self.cluster_tile_tokens)
                )

                task_tile_addr = (fc1_ready_counter.iterator + task_tile_idx).toint()
                flag_tracker = flag_tracker.accumulate(
                    Int32(0), self._flag_batch, task_tile_addr,
                )
                cute.arch.sync_warp()

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.Arrival_Atomic

                phase_bit = phase_bit ^ Int32(1)

                token_idx = token_idx + Int32(num_global_warps)

        # Tail flush: publish any leftover (< self._flag_batch) accumulated release.
        flag_tracker.fire()
        cute.arch.sync_warp()

        if cutlass.const_expr(self.grouped_token_back):
            # Freeze the group table: token-back group reductions may only
            # start once EVERY dispatch warp has appended all of its rows
            # (a group can span experts whose rows belong to other warps).
            # The release-add pairs with the acquire spin at the top of
            # token_back_by_push.
            if lane_idx == Int32(0):
                red_add_release_gpu_s32(
                    dispatch_done_counter.iterator, Int32(1)
                )

        return phase_bit, stored_num_tokens_per_expert

    @cute.jit
    def _adaptive_pace(
        self,
        avg,
        current_window,
        *,
        lo: cutlass.Constexpr[int],
        hi: cutlass.Constexpr[int],
    ):
        # NVLink pacing: EMA the measured round-trip and nanosleep the deviation
        # so outstanding NVLink requests stay bounded and don't head-of-line
        # block this SM's non-NVLink (local) load/store traffic.
        if current_window > avg:
            avg = avg + ((current_window - avg + Int32(3)) // Int32(4))
            sleep_cycle = current_window - avg
            if sleep_cycle > Int32(hi):
                sleep_cycle = Int32(hi)
            if sleep_cycle > Int32(50):
                _nanosleep(sleep_cycle)
        else:
            avg = avg - ((avg - current_window + Int32(3)) // Int32(4))
            sleep_cycle = avg - current_window
            if sleep_cycle > Int32(50):
                _nanosleep(sleep_cycle)
        if avg > Int32(hi):
            avg = Int32(hi)
        if avg < Int32(lo):
            avg = Int32(lo)
        return avg

    @cute.jit
    def grouped_reduce_push(
        self,
        fc2_output_workspace,
        group_rows,
        combine_output,
        combine_sf,
        peer_rank_ptr_mapper,
        smem_ptr_warp,
        grp_gid,
        grp_cnt,
        src_rank,
        src_token,
        lane_idx,
        *,
        local_rank,
    ):
        """Reduce one (src_rank, src_token) group in fp32 and push one row.

        One warp; 512-element chunks (16 per lane).  Member rows come from
        ``group_rows``; the fp32 partial sum is encoded to the combine wire
        format (bf16 pass-through, or per-32 e8m0 + fp8 for the quantized
        formats) in registers, staged through this warp's smem chunk buffer,
        and TMA-pushed into the source rank's ``[t_cap][world_size]`` inbox
        slot for this rank.  Top-k weights were folded into FC1
        (``apply_topk_in_fc1``), so the group sum is a plain add; the source
        side dequantizes back to fp32 and reduces over contributing ranks.
        """
        LANE_ELEMS: cutlass.Constexpr[int] = 16
        CHUNK_ELEMS: cutlass.Constexpr[int] = 32 * LANE_ELEMS
        n_chunks: cutlass.Constexpr[int] = self.hidden // CHUNK_ELEMS
        quantized: cutlass.Constexpr[bool] = self.combine_format.is_quantized
        act_dtype = self.combine_format.act_dtype
        act_bits: cutlass.Constexpr[int] = int(act_dtype.width)
        wire_chunk_bytes: cutlass.Constexpr[int] = CHUNK_ELEMS * act_bits // 8
        lane_wire_bytes: cutlass.Constexpr[int] = LANE_ELEMS * act_bits // 8
        wire_row_bytes: cutlass.Constexpr[int] = self.fc2_token_bytes
        ws_row_bytes: cutlass.Constexpr[int] = self.hidden * 2  # bf16 staging
        slots: cutlass.Constexpr[int] = self.world_size

        ws_base = fc2_output_workspace.iterator.toint()
        smem_base = smem_ptr_warp.toint()

        # Member pool rows (frozen after the dispatch_done barrier).
        member_rows = cute.make_rmem_tensor((self.num_topk,), Int32)
        for m in cutlass.range_constexpr(self.num_topk):
            member_rows[m] = Int32(0)
            if Int32(m) < grp_cnt:
                member_rows[m] = Int32(
                    group_rows[grp_gid * Int32(self.num_topk) + Int32(m)]
                )

        peer_row_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
            combine_output.iterator, src_rank,
        ) + (
            Int64(src_token * Int32(slots) + Int32(local_rank))
            * Int64(wire_row_bytes)
        )

        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128,
        )
        acc = cute.make_rmem_tensor((LANE_ELEMS,), cutlass.Float32)
        term = cute.make_rmem_tensor((LANE_ELEMS,), cutlass.BFloat16)

        for c in cutlass.range(n_chunks, unroll=1):
            lane_elem_base = c * Int32(CHUNK_ELEMS) + lane_idx * Int32(LANE_ELEMS)
            for i in cutlass.range_constexpr(LANE_ELEMS):
                acc[i] = Float32(0.0)
            for m in cutlass.range_constexpr(self.num_topk):
                if Int32(m) < grp_cnt:
                    row_addr = (
                        ws_base
                        + Int64(member_rows[m]) * Int64(ws_row_bytes)
                        + Int64(lane_elem_base) * Int64(2)
                    )
                    src_t = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            row_addr,
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        ),
                        cute.make_layout((LANE_ELEMS,)),
                    )
                    cute.copy(load_atom, src_t, term)
                    acc.store(acc.load() + term.load().to(cutlass.Float32))

            if cutlass.const_expr(quantized):
                # Per-32 e8m0 scale: a scale block spans two adjacent lanes'
                # 16 elements; butterfly-share the amax, encode lane-locally.
                amax = Float32(0.0)
                for i in cutlass.range_constexpr(LANE_ELEMS):
                    v = Float32(acc[i])
                    neg = Float32(0.0) - v
                    amax = cute.arch.fmax(amax, cute.arch.fmax(v, neg))
                partner_amax = Float32(
                    cute.arch.shuffle_sync(amax, lane_idx ^ Int32(1))
                )
                amax = cute.arch.fmax(amax, partner_amax)
                if cutlass.const_expr(act_dtype is cutlass.Float8E4M3FN):
                    rcp_limit: cutlass.Constexpr[float] = Fp8E4M3RcpLimit
                else:
                    rcp_limit: cutlass.Constexpr[float] = Fp8E5M2RcpLimit
                # Round-UP e8m0 in integer bit math (the ue8m0x2 cvt is
                # SM100-only): exponent of amax*rcp_limit, +1 when any
                # mantissa bit is set, satfinite at 254.  amax >= 0, so no
                # sign/NaN handling is needed; the byte IS the e8m0 code.
                fbits = cute.make_rmem_tensor((1,), Float32)
                ibits = cute.recast_tensor(fbits, Int32)
                fbits[0] = amax * Float32(rcp_limit)
                cand_bits = Int32(ibits[0])
                exp_up = (
                    ((cand_bits >> Int32(23)) & Int32(0xFF))
                    + ((cand_bits & Int32(0x7FFFFF)) + Int32(0x7FFFFF))
                    // Int32(0x800000)
                )
                exp_up = cutlass.min(exp_up, Int32(254))
                ibits[0] = exp_up << Int32(23)
                scale_f32 = Float32(fbits[0])
                enc = cute.arch.fmin(
                    cute.arch.rcp_approx(scale_f32), Float32(Fp32Max)
                ) * cute.arch.fmin(
                    scale_f32 * Float32(1e30), Float32(1.0)
                )
                q = cute.make_rmem_tensor((LANE_ELEMS,), act_dtype)
                for i in cutlass.range_constexpr(LANE_ELEMS):
                    q[i] = (Float32(acc[i]) * enc).to(act_dtype)
                smem_q = cute.make_tensor(
                    cute.make_ptr(
                        act_dtype,
                        smem_base + Int64(lane_idx * Int32(lane_wire_bytes)),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout((LANE_ELEMS,)),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), act_dtype,
                        num_bits_per_copy=128,
                    ),
                    q, smem_q,
                )
                # One sf byte per 32-element block: even lanes own their
                # block.  The e8m0 code IS the (round-up) exponent byte, so
                # store it directly -- no fp8 cvt instruction involved.
                if (lane_idx & Int32(1)) == Int32(0):
                    sf_smem = cute.make_tensor(
                        cute.make_ptr(
                            Uint8,
                            smem_base
                            + Int64(wire_chunk_bytes)
                            + Int64(
                                c * Int32(CHUNK_ELEMS // 32)
                                + (lane_idx >> Int32(1))
                            ),
                            cute.AddressSpace.smem,
                            assumed_align=1,
                        ),
                        cute.make_layout((1,)),
                    )
                    sf_smem[0] = Uint8(exp_up)
            else:
                outw = cute.make_rmem_tensor((LANE_ELEMS,), cutlass.BFloat16)
                outw.store(acc.load().to(cutlass.BFloat16))
                smem_o = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        smem_base + Int64(lane_idx * Int32(lane_wire_bytes)),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout((LANE_ELEMS,)),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                        num_bits_per_copy=128,
                    ),
                    outw, smem_o,
                )

            cute.arch.sync_warp()
            cute.arch.fence_proxy("async.shared", space="cta")
            with cute.arch.elect_one():
                tma_store_1d(
                    peer_row_ptr + Int64(c * Int32(wire_chunk_bytes)),
                    smem_ptr_warp,
                    Int32(wire_chunk_bytes),
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
            cute.arch.sync_warp()

        if cutlass.const_expr(quantized):
            # SF plane: hidden/32 e8m0 bytes accumulated behind the data
            # chunk region; one padded-slot push into the source's sf inbox.
            combine_sf_u8g = cute.recast_tensor(combine_sf, Uint8)
            sf_slot_bytes: cutlass.Constexpr[int] = cute.size(
                combine_sf_u8g[0, None, 0].stride
            )
            peer_sf_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
                combine_sf_u8g.iterator, src_rank,
            ) + (
                Int64(src_token * Int32(slots) + Int32(local_rank))
                * Int64(sf_slot_bytes)
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            with cute.arch.elect_one():
                tma_store_1d(
                    peer_sf_ptr,
                    smem_ptr_warp + Int32(wire_chunk_bytes),
                    Int32(self.hidden // 32),
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
            cute.arch.sync_warp()

    @cute.jit
    def token_back_by_push(
        self,
        pull_buffer_ptr,
        pull_mbar_ptr,
        fc2_output_workspace,
        fc2_done_counter,
        token_src_metadata,
        combine_output,
        combine_sf,
        fc2_output_sf,
        group_count,
        group_rows,
        group_done,
        dispatch_done_counter,
        token_back_schedule_counter,
        peer_rank_ptr_mapper,
        phase_bit,
        stored_num_tokens_per_expert,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_sms,
        chunk_bytes: cutlass.Constexpr[int],
    ):
        _iket_emit = (sm_idx == Int32(0)) and (warp_idx == Int32(0))
        avg_token_back_window = Int32(2500)

        # Chunk the fc2 token in ``chunk_bytes`` pieces; the last piece carries
        # the remainder so any chunk_bytes works for any fc2_token_bytes.
        fc2_token_bytes: cutlass.Constexpr[int] = self.fc2_token_bytes
        num_chunks: cutlass.Constexpr[int] = (
            fc2_token_bytes + chunk_bytes - 1
        ) // chunk_bytes
        last_chunk_bytes: cutlass.Constexpr[int] = (
            fc2_token_bytes - (num_chunks - 1) * chunk_bytes
        )

        if cutlass.const_expr(self.push_sf):
            # (token, topk, hidden):(d_topkxhidden, d_hidden, 1)
            combine_sf_u8 = cute.recast_tensor(combine_sf, Uint8)
            sf_token_bytes: cutlass.Constexpr[int] = cute.size(combine_sf_u8[0, None, 0].stride)
            num_sf_chunks: cutlass.Constexpr[int] = (
                sf_token_bytes + chunk_bytes - 1
            ) // chunk_bytes
            last_sf_chunk_bytes: cutlass.Constexpr[int] = (
                sf_token_bytes - (num_sf_chunks - 1) * chunk_bytes
            )

        num_experts_per_lane: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        num_global_warps: cutlass.Constexpr[int] = (
            num_sms * self.num_dispatch_warps
        )

        if cutlass.const_expr(self.grouped_token_back):
            # A group's rows can belong to experts owned by OTHER dispatch
            # warps, so the group table (group_count / group_rows) is only
            # complete once every dispatch warp finished its pull loop.  The
            # acquire spin pairs with each warp's release-add in
            # dispatch_pull's tail.
            spin_wait(
                dispatch_done_counter.iterator,
                lambda v: v >= Int32(num_global_warps),
                fail_sleep_cycles=500,
            )

        schedule_mode = self.token_back_schedule_mode
        atomic_batch = self.token_back_atomic_batch

        # static: stride by the global warp count.  atomic_counter: consume one
        # slot of the current batch, refilling via one grid-scoped
        # atomicAdd(atomic_batch) when exhausted so fast warps keep stealing
        # work.  cuTeDSL forbids closures over enclosing locals -> pass all in.
        def update_token_idx(
            token_idx, batch_remaining, lane_idx, schedule_counter,
            schedule_mode, atomic_batch, num_global_warps,
        ):
            if cutlass.const_expr(schedule_mode == "atomic_counter"):
                batch_remaining = batch_remaining - Int32(1)
                if batch_remaining == Int32(0):
                    base = Int32(0)
                    if lane_idx == Int32(0):
                        base = cute.arch.atomic_add(
                            schedule_counter, Int32(atomic_batch),
                            sem="relaxed", scope="gpu",
                        )
                    token_idx = cute.arch.shuffle_sync(base, Int32(0))
                    batch_remaining = Int32(atomic_batch)
                else:
                    token_idx = token_idx + Int32(1)
            else:
                token_idx = token_idx + Int32(num_global_warps)
            return token_idx, batch_remaining

        if cutlass.const_expr(schedule_mode == "atomic_counter"):
            # Prime the first batch: batch_remaining=1 makes update_token_idx
            # decrement to 0 and pull the initial atomic batch.
            token_idx = Int32(0)
            batch_remaining = Int32(1)
            token_idx, batch_remaining = update_token_idx(
                token_idx, batch_remaining, lane_idx,
                token_back_schedule_counter,
                schedule_mode, atomic_batch, num_global_warps,
            )
        else:
            token_idx = sm_idx * Int32(self.num_dispatch_warps) + warp_idx
            batch_remaining = Int32(0)

        current_expert_idx = Int32(-1)
        confirmed_expert_idx = Int32(-1)
        cur_expert_expected = Int32(0)
        expert_start_idx = Int32(0)
        expert_end_idx = Int32(0)
        expert_pool_block_offset = Int32(0)

        while current_expert_idx < Int32(self.num_experts_per_rank):
            while (token_idx >= expert_end_idx) and (
                current_expert_idx < Int32(self.num_experts_per_rank)
            ):
                prev_valid_count = expert_end_idx - expert_start_idx
                prev_block_count = (
                    prev_valid_count + Int32(self.token_padding_block) - Int32(1)
                ) // Int32(self.token_padding_block)
                expert_pool_block_offset = (
                    expert_pool_block_offset + prev_block_count
                )

                current_expert_idx = current_expert_idx + Int32(1)
                if current_expert_idx < Int32(self.num_experts_per_rank):
                    expert_start_idx = expert_end_idx
                    valid_value = Int32(0)
                    for i in cutlass.range_constexpr(
                        0, num_experts_per_lane, 1
                    ):
                        if current_expert_idx == Int32(
                            i * self.warp_threads
                        ) + lane_idx:
                            valid_value = stored_num_tokens_per_expert[i]
                    total_for_expert = cute.arch.shuffle_sync(
                        valid_value,
                        current_expert_idx % Int32(self.warp_threads),
                    )
                    expert_end_idx = expert_end_idx + total_for_expert

                    cluster_tile_cnt = (
                        total_for_expert
                        + Int32(self.cluster_tile_tokens)
                        - Int32(1)
                    ) // Int32(self.cluster_tile_tokens)
                    # Stash the threshold; the wait is deferred to the expert we
                    # actually land on, so stepped-over experts are never waited.
                    cur_expert_expected = cluster_tile_cnt * Int32(
                        self.fc2_publishes_per_token_cluster_tile
                    )

            if current_expert_idx < Int32(self.num_experts_per_rank):
                # Wait once per processed expert (both indices monotonic; fc2
                # completes in expert order so confirming k implies all < k).
                if current_expert_idx > confirmed_expert_idx:
                    spin_wait(
                        fc2_done_counter.iterator + current_expert_idx,
                        lambda v: v >= cur_expert_expected,
                        fail_sleep_cycles=500,
                    )
                    confirmed_expert_idx = current_expert_idx

                remain_experts = Int32(self.num_experts_per_rank) - current_expert_idx
                token_idx_in_expert = token_idx - expert_start_idx
                pool_token_idx = (
                    expert_pool_block_offset * Int32(self.token_padding_block)
                    + token_idx_in_expert
                )

                md = TokenSrcMetadata.load(
                    token_src_metadata.iterator
                    + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )
                src_rank = md.src_rank
                src_token = md.src_token
                src_topk = md.src_topk
                is_remote_token_back = src_rank != Int32(local_rank)

                smem_ptr_warp = pull_buffer_ptr + warp_idx * Int32(chunk_bytes)
                mbar_ptr_warp = pull_mbar_ptr + warp_idx

                if _iket_emit:
                    _iket.range_push("token_back")
                cute.arch.sync_warp()

                if cutlass.const_expr(self.grouped_token_back):
                    # Combine dedup: the LAST row of a (src_rank, src_token)
                    # group to reach this point pre-reduces every member row
                    # in fp32 and pushes ONE (optionally quantized) row into
                    # the source's [t_cap][world_size] inbox slot keyed by
                    # this rank.  The fence/atomic pair makes every member's
                    # fc2 data visible to the reducer: each member's walk
                    # thread acquire-waited its expert's fc2_done, the fence
                    # releases that view into its group_done add, and the
                    # reducer's fence after observing cnt-1 acquires it.
                    grp_gid = src_rank * Int32(self.max_tokens_per_rank) + src_token
                    cute.arch.fence_acq_rel_gpu()
                    old_done = Int32(0)
                    if lane_idx == Int32(0):
                        old_done = cute.arch.atomic_add(
                            group_done.iterator + grp_gid,
                            Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                    old_done = Int32(
                        cute.arch.shuffle_sync(old_done, Int32(0))
                    )
                    grp_cnt = Int32(group_count[grp_gid])
                    if old_done == grp_cnt - Int32(1):
                        cute.arch.fence_acq_rel_gpu()
                        self.grouped_reduce_push(
                            fc2_output_workspace,
                            group_rows,
                            combine_output,
                            combine_sf,
                            peer_rank_ptr_mapper,
                            smem_ptr_warp,
                            grp_gid,
                            grp_cnt,
                            src_rank,
                            src_token,
                            lane_idx,
                            local_rank=local_rank,
                        )

                # DATA plane: only the dispatch DATA path pushes here; epi_warps
                # has the epilogue STG/UBLK the data straight to the peer.
                if cutlass.const_expr(self.push_data and not self.grouped_token_back):
                    local_token_addr = (
                        fc2_output_workspace.iterator.toint()
                        + Int64(pool_token_idx) * Int64(fc2_token_bytes)
                    )
                    peer_combine_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
                        combine_output.iterator, src_rank,
                    )
                    if cutlass.const_expr(self.token_back_reduce_topk):
                        peer_token_offset = Int64(src_token) * Int64(fc2_token_bytes)
                    else:
                        peer_token_offset = (
                            Int64(src_token * Int32(self.num_topk) + src_topk)
                            * Int64(fc2_token_bytes)
                        )
                    peer_token_ptr = peer_combine_ptr + peer_token_offset

                    for chunk in cutlass.range(num_chunks, unroll=1):
                        t0 = read_clock64()
                        chunk_off = Int64(chunk * chunk_bytes)
                        peer_chunk_ptr = peer_token_ptr + chunk_off

                        this_bytes = Int32(chunk_bytes)
                        if cutlass.const_expr(last_chunk_bytes != chunk_bytes):
                            if chunk == Int32(num_chunks - 1):
                                this_bytes = Int32(last_chunk_bytes)

                        with cute.arch.elect_one():
                            tma_load_1d_raw(
                                smem_ptr_warp,
                                local_token_addr + chunk_off,
                                mbar_ptr_warp,
                                this_bytes,
                            )
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_ptr_warp, this_bytes,
                            )
                            cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                            if cutlass.const_expr(self.token_back_reduce_topk):
                                cp_reduce_async_bulk_add_noftz_bf16_s2g(
                                    peer_chunk_ptr,
                                    smem_ptr_warp,
                                    this_bytes,
                                )
                            else:
                                tma_store_1d(
                                    peer_chunk_ptr,
                                    smem_ptr_warp,
                                    this_bytes,
                                )
                        phase_bit = phase_bit ^ Int32(1)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                        t1 = read_clock64()
                        current_window = Int32(t1 - t0)
                        if is_remote_token_back and remain_experts > Int32(4):
                            avg_token_back_window = self._adaptive_pace(
                                avg_token_back_window, current_window, lo=1000, hi=5000,
                            )

                if cutlass.const_expr(self.push_sf and not self.grouped_token_back):
                    sf_local_addr = fc2_output_sf[pool_token_idx, 0, None].iterator.toint()
                    sf_peer_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
                        combine_sf_u8[src_token, src_topk, None].iterator, src_rank,
                    )
                    for chunk in cutlass.range(num_sf_chunks, unroll=1):
                        t0 = read_clock64()
                        chunk_off = Int64(chunk * chunk_bytes)
                        this_bytes = Int32(chunk_bytes)
                        if cutlass.const_expr(last_sf_chunk_bytes != chunk_bytes):
                            if chunk == Int32(num_sf_chunks - 1):
                                this_bytes = Int32(last_sf_chunk_bytes)
                        with cute.arch.elect_one():
                            tma_load_1d_raw(
                                smem_ptr_warp,
                                sf_local_addr + chunk_off,
                                mbar_ptr_warp,
                                this_bytes,
                            )
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_ptr_warp, this_bytes,
                            )
                            cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                            tma_store_1d(
                                sf_peer_ptr + chunk_off,
                                smem_ptr_warp,
                                this_bytes,
                            )
                        phase_bit = phase_bit ^ Int32(1)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                        if is_remote_token_back:
                            round_trip = Int32(read_clock64() - t0)
                            if round_trip <= this_bytes * Int32(5) // Int32(4):
                                _nanosleep(this_bytes // Int32(4))

                if _iket_emit:
                    _iket.range_pop()

                token_idx, batch_remaining = update_token_idx(
                    token_idx, batch_remaining, lane_idx,
                    token_back_schedule_counter,
                    schedule_mode, atomic_batch, num_global_warps,
                )

        cute.arch.fence_acq_rel_sys()
        # _fence_rel_sys()

    @cute.jit
    def nvlink_barrier(
        self,
        nvlink_barrier_signal,
        nvlink_barrier_counter,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
        prologue_grid_sync: cutlass.Constexpr[bool],
        epilogue_grid_sync: cutlass.Constexpr[bool],
    ):
        # software_grid_sync expects a dispatch-group-relative thread id.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

        if prologue_grid_sync:
            software_grid_sync(grid_sync_counter, sm_idx, num_sms, tid_in_group,
                               num_threads=self.num_dispatch_threads)

        if sm_idx == 0:
            if warp_idx == 0:
                # Sense-reversing ping-pong barrier. The low 2 bits of the counter
                # pick the signal slot (phase 0/1) and the direction (+1 up to
                # world_size, then -1 back to 0), so the two slots self-cancel over
                # a 4-call cycle and never need an explicit reset -- required
                # because the signal is symmetric peer memory that ncu kernel
                # replay cannot snapshot/restore.
                status = nvlink_barrier_counter[0] & Int32(3)
                signal_phase = status & Int32(1)
                signal_sign = status >> Int32(1)
                signal_delta = Int32(1)
                target = Int32(self.world_size)
                if signal_sign != Int32(0):
                    signal_delta = Int32(-1)
                    target = Int32(0)

                nbs_local_base = nvlink_barrier_signal.iterator.toint()
                if lane_idx < Int32(self.world_size):
                    lane_peer_addr = peer_rank_ptr_mapper.map(
                        nbs_local_base, lane_idx,
                        Int64(signal_phase * Int32(4)),
                    )
                    red_add_release_sys_s32_raw(lane_peer_addr, signal_delta)
                cute.arch.sync_warp()

                if lane_idx == 0:
                    cute.arch.atomic_add(
                        nvlink_barrier_counter.iterator,
                        Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    local_signal_ptr = nvlink_barrier_signal.iterator + signal_phase
                    while cute.arch.load(local_signal_ptr, Int32, sem="acquire", scope="sys") != target:
                        pass

        if epilogue_grid_sync:
            software_grid_sync(grid_sync_counter, sm_idx, num_sms, tid_in_group,
                               num_threads=self.num_dispatch_threads)

    @cute.jit
    def dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_linear_id = (
            Int32(bidx)
            + Int32(self.cluster_shape_mn[0]) * Int32(bidy)
            + Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0])
            * Int32(bidz)
        )
        local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)

        # Record all four dispatch warps in CTA 0. Recording every persistent
        # CTA duplicates the same role and makes PIC-C's trace buffer too large.
        iket_active = cta_linear_id == Int32(0)
        if iket_active:
            _iket.range_push("Dispatch_Prep")

        self.dispatch_prep(
            token_comm_storage,
            token_comm_args.topk_idx,
            token_comm_args.expert_send_count,
            token_comm_args.src_token_topk_idx,
            token_comm_args.token_rank_mask,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            local_rank=token_comm_args.local_rank,
            num_tokens=token_comm_args.input_token_buffer.shape[0],
            num_sms=token_comm_args.sm_count,
        )

        if iket_active:
            _iket.range_pop()
            _iket.range_push("Dispatch_Barrier")

        self.dispatch_barrier(
            token_comm_args.expert_send_count,
            token_comm_args.expert_recv_count,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.nvlink_barrier_signal,
            token_comm_args.grid_sync_counter,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            local_rank=token_comm_args.local_rank,
            num_sms=token_comm_args.sm_count,
            nvlink_barrier_counter=token_comm_args.nvlink_barrier_counter,
        )

        nb_dispatch_to_sched = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb_dispatch_to_sched.arrive()

        if iket_active:
            _iket.range_pop()
            _iket.range_push("Dispatch_Pull")

        phase_bit, stored_num_tokens_per_expert = self.dispatch_pull(
            token_comm_storage,
            token_comm_args.input_token_buffer,
            token_comm_args.input_sf_buffer,
            token_comm_args.input_topk_weights_buffer,
            token_comm_args.src_token_topk_idx,
            token_comm_args.expert_recv_count,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.fc1_input_token_buffer,
            token_comm_args.fc1_input_sf_buffer,
            token_comm_args.fc1_input_topk_weights_buffer,
            token_comm_args.fc1_ready_counter,
            token_comm_args.token_src_metadata,
            token_comm_args.carrier_row_table,
            token_comm_args.group_count,
            token_comm_args.group_rows,
            token_comm_args.dispatch_done_counter,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
        )

        if iket_active:
            _iket.range_pop()

        if cutlass.const_expr(self.enable_token_back and not self.token_back_standalone):
            if iket_active:
                _iket.range_push("Token_Back_By_Push")

            self.token_back_by_push(
                token_comm_storage.pull_buffer.data_ptr(),
                token_comm_storage.pull_mbar.data_ptr(),
                token_comm_args.fc2_output_workspace,
                token_comm_args.fc2_done_counter,
                token_comm_args.token_src_metadata,
                token_comm_args.combine_output,
                token_comm_args.combine_sf,
                token_comm_args.fc2_output_sf,
                token_comm_args.group_count,
                token_comm_args.group_rows,
                token_comm_args.group_done,
                token_comm_args.dispatch_done_counter,
                token_comm_args.token_back_schedule_counter,
                token_comm_args.peer_rank_ptr_mapper,
                phase_bit,
                stored_num_tokens_per_expert,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                local_rank=token_comm_args.local_rank,
                num_sms=token_comm_args.sm_count,
                chunk_bytes=self.hidden_bytes,
            )

            if iket_active:
                _iket.range_pop()

    @cute.jit
    def token_back_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_linear_id = (
            Int32(bidx)
            + Int32(self.cluster_shape_mn[0]) * Int32(bidy)
            + Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0])
            * Int32(bidz)
        )
        local_warp_idx = Int32(warp_idx) - Int32(self.token_back_warp_start)

        # Handshake: dispatch_barrier done => expert_recv_count_sum populated.
        nb_dispatch_to_sched = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb_dispatch_to_sched.arrive_and_wait()

        tb_pull_mbar_ptr = token_comm_storage.tb_pull_mbar.data_ptr()
        tb_pull_buffer_ptr = token_comm_storage.tb_pull_buffer.data_ptr()
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(tb_pull_mbar_ptr + local_warp_idx, 1)
        cute.arch.sync_warp()

        NUM_EXPERTS_PER_LANE: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        stored_num_tokens_per_expert = []
        for _ in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            stored_num_tokens_per_expert.append(Int32(0))
        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            e_idx_for_lane = Int32(i * self.warp_threads) + lane_idx
            if e_idx_for_lane < Int32(self.num_experts_per_rank):
                sum_packed_init = token_comm_args.expert_recv_count_sum[e_idx_for_lane]
                stored_num_tokens_per_expert[i] = Int32(
                    Int64(sum_packed_init) & Int64(0xFFFFFFFF)
                )
        cute.arch.sync_warp()

        iket_active = cta_linear_id == Int32(0)
        if iket_active:
            _iket.range_push("Token_Back_By_Push_Standalone")

        self.token_back_by_push(
            tb_pull_buffer_ptr,
            tb_pull_mbar_ptr,
            token_comm_args.fc2_output_workspace,
            token_comm_args.fc2_done_counter,
            token_comm_args.token_src_metadata,
            token_comm_args.combine_output,
            token_comm_args.combine_sf,
            token_comm_args.fc2_output_sf,
            token_comm_args.group_count,
            token_comm_args.group_rows,
            token_comm_args.group_done,
            token_comm_args.dispatch_done_counter,
            token_comm_args.token_back_schedule_counter,
            token_comm_args.peer_rank_ptr_mapper,
            Int32(0),
            stored_num_tokens_per_expert,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            local_rank=token_comm_args.local_rank,
            num_sms=token_comm_args.sm_count,
            chunk_bytes=self.tb_chunk_bytes,
        )

        if iket_active:
            _iket.range_pop()

    @cute.jit
    def tail_reset_counters(
        self,
        token_comm_args,
        target_zero_tensor,
        *,
        cta_linear_id,
        local_warp_idx,
        lane_idx,
    ):
        """Per-lane 4B (Int32) bulk-zero of one accumulating-counter prefix.

        ``target_zero_tensor`` is an Int32 view over a workspace's front counter
        region (megamoe_kernel front-places every counter that must restart at 0
        each launch; data buffers and the phase-flip ``nvlink_barrier_signal`` sit
        after the prefix and are untouched). The zeroing is spread across all
        dispatch threads grid-wide. kernel_tail calls this twice:
          * SHARED prefix (expert_recv_count / _sum) BETWEEN the two nvlink
            barriers, so the final barrier publishes the zeros cross-rank -- needed
            when the next launch is another MegaMoE reusing the shared workspace
            with no intervening rank sync;
          * LOCAL prefix (l1_arrival / expert_send / grid_sync / nvlink_barrier /
            fc1_done [+ fc2_done / token_back_schedule / load_balance]) AFTER the
            last barrier -- rank-local (next kernel sees it via stream order) and
            grid_sync/nvlink barrier counters stay live until that last barrier.
        Only the FIRST launch relies on a caller-zeroed workspace.
        """
        thread_linear = (
            (cta_linear_id * Int32(self.num_dispatch_warps) + local_warp_idx)
            * Int32(self.warp_threads)
            + lane_idx
        )
        stride = Int32(token_comm_args.sm_count * self.num_dispatch_threads)

        count = cute.size(target_zero_tensor)
        i = thread_linear
        while i < Int32(count):
            target_zero_tensor[i] = Int32(0)
            i = i + stride

    @cute.jit
    def kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        nb_kernel_tail = pipeline.NamedBarrier(
            barrier_id=self.kernel_tail_named_barrier_id,
            num_threads=self.kernel_tail_threads,
        )
        nb_kernel_tail.arrive_and_wait()

        # Only the dispatch warps run NVLink cleanup; standalone token-back
        # warps (>= token_back_warp_start) just join the rendezvous above.
        if (warp_idx >= self.dispatch_warp_start) and (
            warp_idx < self.dispatch_warp_start + self.num_dispatch_warps
        ):
            bidx, bidy, bidz = cute.arch.block_idx()
            cta_linear_id = (
                Int32(bidx)
                + Int32(self.cluster_shape_mn[0]) * Int32(bidy)
                + Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0])
                * Int32(bidz)
            )
            local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)
            # Per-launch nvlink barrier count must be a multiple of 4 so the
            # sense-reversing signal self-cancels back to its start state. The
            # launch already does 1 (dispatch_barrier) + 2 below (drain + publish,
            # around the shared reset) = 3. Under ncu kernel replay the signal is
            # cross-device and can't be restored across passes, so pad with one
            # extra drain to reach 4. Non-ncu rides the phase counter across launch
            # boundaries and needs only 3.
            #
            # MEGA_USE_NCU is a dev-only profiling hack -- read inline here on
            # purpose: it is never exposed to customers and the ncu path is not
            # kept in production, so it must NOT become a constructor field or a
            # kernel-name constexpr.
            if cutlass.const_expr(os.environ.get("MEGA_USE_NCU", "0") == "1"):
                self.nvlink_barrier(
                    token_comm_args.nvlink_barrier_signal,
                    token_comm_args.nvlink_barrier_counter,
                    token_comm_args.grid_sync_counter,
                    token_comm_args.peer_rank_ptr_mapper,
                    cta_linear_id,
                    local_warp_idx,
                    lane_idx,
                    num_sms=token_comm_args.sm_count,
                    prologue_grid_sync=True,
                    epilogue_grid_sync=True,
                )
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                num_sms=token_comm_args.sm_count,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
            # Shared counters between the barriers: the slot=0 barrier below
            # publishes these zeros cross-rank for a back-to-back MegaMoE relaunch.
            self.tail_reset_counters(
                token_comm_args,
                token_comm_args.shared_zero_prefix,
                cta_linear_id=cta_linear_id,
                local_warp_idx=local_warp_idx,
                lane_idx=lane_idx,
            )
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                num_sms=token_comm_args.sm_count,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
            # Local counters last: rank-local, and grid_sync/nvlink_barrier
            # counters above stay live until this final barrier completes.
            self.tail_reset_counters(
                token_comm_args,
                token_comm_args.local_zero_prefix,
                cta_linear_id=cta_linear_id,
                local_warp_idx=local_warp_idx,
                lane_idx=lane_idx,
            )
