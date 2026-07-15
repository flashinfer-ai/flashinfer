"""Token communication implementations for MegaMoE-style kernels.

Current implementation: token-in pull with token-back push.  The standalone
``dispatch_kernel`` uses the same object methods as the fused MegaMoE kernel.
"""

import dataclasses
import os
from typing import Any, ClassVar, Dict, List, Literal, Union

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
    red_add_release_sys_s32_raw,
    red_add_release_sys_u64_raw,  # noqa: F401
    stg_b8_raw,
    stg_b32_raw,
    stg_b64_raw,
    tma_load_1d_raw,
    tma_store_1d,
    _fence_rel_sys,  # noqa: F401
)
from .flag_batch import GpuReleaseFlagBatchTracker
from .sf_swizzle import sf_atom_int32_offset
from cutlass._mlir import ir
from moe_nvfp4_swapab.moe_utils import _nanosleep, spin_wait
from common.moe_utils import quant_sfd_row, cvt_f32x4_to_f8x4_pack_i32  # noqa: F401
from cutlass._mlir.dialects import math as _cute_math


# --- perf probe (branch probe/no-combine-store) -----------------------------
# Drop the combine-phase cross-rank TMA store + its NVLink pacing to isolate the
# NVLink write-out cost.  The reuse_dispatch_warps pipeline is otherwise intact
# (epilogue STG to local workspace, dispatch warps still spin_wait / read
# metadata / TMA-load the local fc2 token into smem / schedule).  Output becomes
# garbage (combine_output never written) -> MUST run with --skip_ref_check.
# Toggle via env MEGA_NO_COMBINE_STORE=1.  Gated by cutlass.const_expr so both
# the remote store and the _adaptive_pace nanosleep are eliminated at trace time
# when enabled (zero residual instructions; baseline path byte-identical).
_MEGA_NO_COMBINE_STORE = os.environ.get("MEGA_NO_COMBINE_STORE", "0") == "1"
if _MEGA_NO_COMBINE_STORE:
    print(
        "[MEGA] PROBE: combine remote TMA store + NVLink pacing DISABLED "
        "(perf probe; combine_output not written, use --skip_ref_check)",
        flush=True,
    )

# Cost-decomposition probes for the mxfp8 quantize-on-push hot path
# (_quant_chunk_mxfp8).  Both const_expr-gated -> zero residual when off.
#   NO_SF_STORE  : skip the per-block 1-byte remote e8m0 scale stores (stg_b8_raw)
#                  -> isolates the cost of the 44M tiny remote st.global.u8.
#   SKIP_ARITH   : replace amax/scale/clamp with a single direct .to(fp8) of the
#                  raw loaded values -> isolates the per-element scalar arithmetic.
# Perf probes only (output is garbage) -> run with --skip_ref_check.
_MEGA_QUANT_NO_SF_STORE = os.environ.get("MEGA_QUANT_NO_SF_STORE", "0") == "1"
_MEGA_QUANT_SKIP_ARITH = os.environ.get("MEGA_QUANT_SKIP_ARITH", "0") == "1"
if _MEGA_QUANT_NO_SF_STORE or _MEGA_QUANT_SKIP_ARITH:
    print(
        f"[MEGA] PROBE: quant decomposition NO_SF_STORE={_MEGA_QUANT_NO_SF_STORE} "
        f"SKIP_ARITH={_MEGA_QUANT_SKIP_ARITH} (perf probe, use --skip_ref_check)",
        flush=True,
    )


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
    "combine_output",
    "combine_output_q",
    "combine_sf_q",
    "combine_global_q",
    "fc2_output_workspace",
    "fc2_output_sf_workspace",
    "fc2_output_global_workspace",
    "fc2_done_counter",
    "token_back_schedule_counter",
    "nvlink_barrier_signal",
    "nvlink_barrier_counter",
    "grid_sync_counter",
    "peer_rank_ptr_mapper",
)

_CONST_FIELDS = (
    "world_size",
    "local_rank",
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
        combine_output_q: cute.Tensor = None,
        combine_sf_q: cute.Tensor = None,
        combine_global_q: cute.Tensor = None,
        nvlink_barrier_counter: cute.Tensor,
        grid_sync_counter: cute.Tensor,
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
        fc2_output_sf_workspace: cute.Tensor = None,
        fc2_output_global_workspace: cute.Tensor = None,
        fc2_done_counter: cute.Tensor = None,
        token_back_schedule_counter: cute.Pointer = None,
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
        self.combine_output = combine_output
        self.combine_output_q = combine_output_q
        self.combine_sf_q = combine_sf_q
        # nvfp4 only: per-32 fp32 global scale plane (peer-symmetric).
        self.combine_global_q = combine_global_q
        self.fc2_output_workspace = fc2_output_workspace
        # local scale plane(s) the fc2 epilogue writes alongside the
        # packed-data workspace.  mxfp8: e8m0 (pool,1,hidden//32) u8.  nvfp4:
        # e4m3 sfc (pool,1,hidden//16) u8 here + an fp32 global plane below.
        self.fc2_output_sf_workspace = fc2_output_sf_workspace
        # nvfp4 only: local per-32 fp32 global plane (pool,1,hidden//32).
        self.fc2_output_global_workspace = fc2_output_global_workspace
        self.fc2_done_counter = fc2_done_counter
        self.token_back_schedule_counter = token_back_schedule_counter
        self.nvlink_barrier_signal = nvlink_barrier_signal
        self.nvlink_barrier_counter = nvlink_barrier_counter
        self.grid_sync_counter = grid_sync_counter
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

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "TokenCommArgs":
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
        local_rank: int,
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
        fc2_output_dtype=None,
        fc2_publishes_per_token_cluster_tile: int = 0,
        token_back_reduce_topk: bool = False,
        token_back_standalone: bool = False,
        flag_batch: int = 1,
        is_swap_ab: bool = False,
        token_back_schedule_mode: Literal["static", "atomic_counter"] = "static",
        fc2_output_quant_mode: Literal["bf16", "mxfp8", "nvfp4"] = "bf16",
    ) -> None:
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.num_experts_per_rank = num_experts_per_rank
        self.num_total_experts = num_total_experts
        self.hidden = hidden
        self.fc1_token_dtype = fc1_token_dtype
        self.hidden_bytes = hidden * int(fc1_token_dtype.width) // 8
        self.sf_uint32_per_token = sf_uint32_per_token
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.cluster_tile_tokens = cluster_tile_tokens
        self.cluster_shape_mn = cluster_shape_mn
        if flag_batch < 1 or flag_batch > 32:
            raise ValueError(f"flag_batch must be in [1, 32], got {flag_batch}.")
        # Release-flag batch size consumed by dispatch_pull as a Python int.
        # One warp lane carries one delayed release target.
        self._flag_batch = flag_batch
        self.is_swap_ab = is_swap_ab

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

        # Standalone token-back: a dedicated warpgroup (size == dispatch group)
        # right after the dispatch warps, active only when token-back is enabled.
        # It joins the dispatch->sched handshake and the kernel-tail rendezvous.
        self.token_back_standalone = (
            fc2_output_dtype is not None
        ) and token_back_standalone
        self.num_token_back_warps = (
            self.num_dispatch_warps if self.token_back_standalone else 0
        )
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

        self.fc2_output_dtype = fc2_output_dtype
        if token_back_reduce_topk:
            if fc2_output_dtype is None:
                raise ValueError(
                    "token_back_reduce_topk=True requires fc2_output_dtype "
                    "to enable token-back."
                )
            if fc2_output_dtype is not cutlass.BFloat16:
                raise NotImplementedError(
                    "token_back_reduce_topk currently supports BF16 fc2 "
                    f"output only, got {fc2_output_dtype}."
                )
        self.token_back_reduce_topk = token_back_reduce_topk
        if fc2_output_dtype is not None:
            self.fc2_token_bytes = hidden * int(fc2_output_dtype.width) // 8
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
                    "fc2_output_dtype is set (token_back_by_push enabled)."
                )
            self.fc2_publishes_per_token_cluster_tile = (
                fc2_publishes_per_token_cluster_tile
            )
        else:
            self.fc2_token_bytes = 0
            self.fc2_num_chunks = 0
            self.fc2_publishes_per_token_cluster_tile = 0

        # Architecture B (in-kernel quantize-on-push): the fc2 epilogue keeps
        # writing bf16 to the LOCAL workspace (so fc2_token_bytes / fc2_num_chunks
        # above stay bf16-sized for the local TMA load), but the cross-rank push
        # quantizes each 32-element block on the fly and writes a SMALLER wire
        # format to combine_output_q (+ scale to combine_sf_q).  These widths
        # decouple the peer-store stride from the local-read stride.
        if fc2_output_quant_mode not in ("bf16", "mxfp8", "nvfp4"):
            raise ValueError(
                f"fc2_output_quant_mode must be 'bf16', 'mxfp8', or 'nvfp4'; "
                f"got {fc2_output_quant_mode!r}."
            )
        self.fc2_output_quant_mode = fc2_output_quant_mode
        self.combine_is_quantized = (
            fc2_output_dtype is not None and fc2_output_quant_mode != "bf16"
        )
        if self.combine_is_quantized:
            if token_back_reduce_topk:
                raise NotImplementedError(
                    "quantized combine (mxfp8/nvfp4) is a form-A path; "
                    "token_back_reduce_topk must be False (no fp8/fp4 cp.reduce)."
                )
            self.quant_block = 32
            if hidden % self.quant_block != 0:
                raise ValueError(
                    f"quantized combine needs hidden ({hidden}) divisible by "
                    f"block={self.quant_block}."
                )
            # Per-token-cell wire bytes on the symmetric heap:
            #   mxfp8: hidden fp8 (1 B/elem) + hidden/32 e8m0 (1 B/block)
            #   nvfp4 (tile-wise two-level): hidden/2 fp4 (2 elem/B)
            #          + hidden/16 e4m3 sfc (1 B/16-block)
            #          + hidden/32 fp32 global (4 B/32-tile)  -> a THIRD plane.
            self.combine_has_global = fc2_output_quant_mode == "nvfp4"
            if fc2_output_quant_mode == "mxfp8":
                self.combine_q_token_bytes = hidden
                self.combine_sf_token_bytes = hidden // self.quant_block
                self.combine_global_token_bytes = 0
            else:  # nvfp4
                self.combine_q_token_bytes = hidden // 2
                self.combine_sf_token_bytes = hidden // 16  # per-16 e4m3 sfc
                # PACK6 combined SF plane: 8 B per 32-elem tile, INTERLEAVED
                # [ global fp32 (4B) | sfc0 (1B) | sfc1 (1B) | pad (2B) ], N=hidden//
                # quant_block tiles.  The per-16 sfc is packed in here too, so the
                # separate sfc plane is NOT pushed for nvfp4 (token-back pushes only
                # data + this combined plane).  Must match the epilogue b64 store, the
                # global_ws 8N region, and the mega_runner combine_global_q 8N alloc.
                self.combine_global_token_bytes = (hidden // self.quant_block) * 8
            # Local bf16 chunk (chunk_bytes == hidden_bytes) holds this many
            # elements; its quantized footprint is the per-chunk store size.
            self.quant_chunk_elems = self.hidden_bytes // 2  # bf16 -> elems
            if self.combine_q_token_bytes % self.fc2_num_chunks != 0:
                raise ValueError(
                    "combine_q_token_bytes must split evenly across "
                    f"fc2_num_chunks={self.fc2_num_chunks}."
                )
            self.combine_q_chunk_bytes = (
                self.combine_q_token_bytes // self.fc2_num_chunks
            )
            self.combine_sf_chunk_blocks = self.quant_chunk_elems // self.quant_block
        else:
            self.quant_block = 0
            self.combine_has_global = False
            self.combine_q_token_bytes = 0
            self.combine_sf_token_bytes = 0
            self.combine_global_token_bytes = 0
            self.quant_chunk_elems = 0
            self.combine_q_chunk_bytes = 0
            self.combine_sf_chunk_blocks = 0

    @property
    def enable_token_back(self) -> bool:
        return self.fc2_output_dtype is not None

    def extra_smem_storage_class(self) -> type:
        hidden_bytes = self.hidden_bytes
        num_total_experts = self.num_total_experts

        if self.token_back_standalone:

            @cute.struct
            class TokenCommStorage:
                pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
                smem_expert_count: cute.struct.MemRange[Int32, num_total_experts]
                pull_buffer: cute.struct.Align[
                    cute.struct.MemRange[Uint8, self.num_dispatch_warps * hidden_bytes],
                    16,
                ]
                tb_pull_mbar: cute.struct.MemRange[Int64, self.num_token_back_warps]
                tb_pull_buffer: cute.struct.Align[
                    cute.struct.MemRange[
                        Uint8, self.num_token_back_warps * self.tb_chunk_bytes
                    ],
                    16,
                ]

            return TokenCommStorage

        # the quantized combine no longer needs per-warp quant scratch
        # -- the fc2 epilogue does the quant, and token-back is a pure byte-copy
        # staged through the existing pull_buffer.  So the storage layout is the
        # same as the bf16 path (dropping q_scratch/sf_scratch keeps smem under
        # the ~228KB cap that the doubled chunk size would otherwise blow).
        @cute.struct
        class TokenCommStorage:  # type: ignore[no-redef]
            pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
            smem_expert_count: cute.struct.MemRange[Int32, num_total_experts]
            pull_buffer: cute.struct.Align[
                cute.struct.MemRange[Uint8, self.num_dispatch_warps * hidden_bytes], 16
            ]

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
            counter_slot = (
                work_tile_info.cumulative_token_block_count + work_tile_info.tile_n_idx
            )
            peek_threshold = work_tile_info.valid_tokens_in_cta_tile
        else:
            counter_slot = (
                work_tile_info.cumulative_token_block_count
                + work_tile_info.tile_m_idx // cutlass.Int32(self.cluster_shape_mn[0])
            )
            peek_threshold = work_tile_info.valid_tokens_in_cluster_tile

        counter_ptr = token_comm_args.fc1_ready_counter.iterator + counter_slot
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
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
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
        num_dispatch_warps_per_grid: cutlass.Constexpr[int] = (
            num_sms * self.num_dispatch_warps
        )

        base_token_for_warp = (
            sm_idx * self.num_dispatch_warps + warp_idx
        ) * tokens_per_warp
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
            0,
            self.num_total_experts,
            self.experts_per_dispatch_pass,
        ):
            expert_id = Int32(offset + warp_idx * self.warp_threads + lane_idx)
            if expert_id < Int32(self.num_total_experts):
                slot_ptr = smem_count_ptr + expert_id
                local_count = (slot_ptr).load()
                delta = (Int64(1) << Int64(32)) | (
                    Int64(local_count) & Int64(0xFFFFFFFF)
                )
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
                        (local_expert * Int32(self.world_size) + Int32(self.local_rank))
                        * Int32(MAX_SLOT_C)
                        + slot
                    ) * Int32(4)
                    peer_addr = peer_rank_ptr_mapper.map(
                        src_token_topk_idx.iterator.toint(),
                        dst_rank,
                        Int64(elem_off),
                    )
                    stg_b32_raw(peer_addr, token_topk_word)
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
        num_sms,
        nvlink_barrier_counter=None,
    ):
        # software_grid_sync expects a dispatch-group-relative thread id.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

        software_grid_sync(
            grid_sync_counter,
            sm_idx,
            num_sms,
            tid_in_group,
            num_threads=self.num_dispatch_threads,
        )

        if sm_idx == 0:
            for offset in cutlass.range_constexpr(
                0,
                self.num_total_experts,
                self.experts_per_dispatch_pass,
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
                        Int32(self.local_rank) * Int32(self.num_experts_per_rank)
                        + dst_local_expert
                    ) * Int32(8)
                    erc_peer_addr = peer_rank_ptr_mapper.map(
                        erc_local_base,
                        dst_rank,
                        Int64(erc_elem_off),
                    )
                    stg_b64_raw(erc_peer_addr, Int64(token_count_u32))
                    ercs_local_base = expert_recv_count_sum.iterator.toint()
                    ercs_peer_addr = peer_rank_ptr_mapper.map(
                        ercs_local_base,
                        dst_rank,
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
            slot=0,
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
            (sm_idx == Int32(0)) and (warp_idx == Int32(0)) and (lane_idx == Int32(0))
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
                expert_pool_block_offset = expert_pool_block_offset + prev_block_count
                # Mirror cumul for the release-counter granularity (self.cluster_tile_tokens).
                prev_task_tile_count = (
                    prev_valid_count + Int32(self.cluster_tile_tokens) - Int32(1)
                ) // Int32(self.cluster_tile_tokens)
                expert_task_tile_offset = expert_task_tile_offset + prev_task_tile_count
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
                    for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
                        if (
                            current_expert_idx
                            == Int32(i * self.warp_threads) + lane_idx
                        ):
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
                        length = Int32(cute.arch.warp_redux_sync(v_for_min, "min"))

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
                src_token = Int32(src_token_topk // Uint32(self.num_topk))
                src_topk = Int32(src_token_topk % Uint32(self.num_topk))

                cur_peer_offset = peer_rank_ptr_mapper.map(
                    Int64(0), current_rank_in_expert_idx, Int64(0)
                )
                inp_tok_local_base = input_token_buffer.iterator.toint()
                inp_sf_local_base = input_sf_buffer.iterator.toint()
                inp_w_local_base = input_topk_weights_buffer.iterator.toint()

                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes)
                    )
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
                    _iket.range_push("Pull.Weight_LDG")  # (= ST phase)

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        sf_int32_pos = sf_atom_int32_offset(
                            sf_token_in_pool_axis,
                            j,
                            num_k_atoms=self.sf_uint32_per_token,
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

                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)

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
                    Int32(0),
                    self._flag_batch,
                    task_tile_addr,
                )
                cute.arch.sync_warp()

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.Arrival_Atomic

                phase_bit = phase_bit ^ Int32(1)

                token_idx = token_idx + Int32(num_global_warps)

        # Tail flush: publish any leftover (< self._flag_batch) accumulated release.
        flag_tracker.fire()
        cute.arch.sync_warp()

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
    def _quant_chunk_mxfp8(
        self,
        smem_bf16_ptr,
        q_smem_ptr,
        peer_sf_addr,
        chunk_idx,
        lane_idx,
    ):
        """Warp-cooperative MXFP8 quantize of one bf16 chunk in SMEM.

        Reads ``quant_chunk_elems`` bf16 from ``smem_bf16_ptr`` (this warp's
        pull buffer), quantizes each 32-element block to fp8_e4m3 + one e8m0
        scale, writes packed fp8 to ``q_smem_ptr`` (this warp's chunk scratch),
        and stores the e8m0 scale byte DIRECTLY to peer gmem at
        ``peer_sf_addr + chunk_idx*blocks + blk`` via a genuine 1-byte store
        (a 1-element SMEM autovec_copy would widen to ~32 bits and let the 32
        lanes clobber each other's adjacent scale bytes).  The 32 lanes split
        the blocks (each block independent).
        """
        nblk: cutlass.Constexpr = self.combine_sf_chunk_blocks
        blk: cutlass.Constexpr = self.quant_block
        nrounds: cutlass.Constexpr = (nblk + self.warp_threads - 1) // self.warp_threads

        s_bf16 = cute.make_tensor(
            cute.recast_ptr(smem_bf16_ptr, dtype=cutlass.BFloat16),
            cute.make_layout(self.quant_chunk_elems),
        )
        # fp8 destination viewed as Int32 words (4 fp8 / word): the packed
        # cvt.rn.satfinite.e4m3x2.f32 emits i32 words, and the whole 32-elem
        # block lands as one vectorized 8xInt32 (=32 fp8, 16 B aligned) copy.
        s_q_i32 = cute.make_tensor(
            cute.recast_ptr(q_smem_ptr, dtype=cutlass.Int32),
            cute.make_layout(self.quant_chunk_elems // 4),
        )
        nqw: cutlass.Constexpr = blk // 4
        sf_chunk_base = chunk_idx * Int32(nblk)

        for r in cutlass.range_constexpr(nrounds):
            blk_idx = Int32(r * self.warp_threads) + lane_idx
            if blk_idx < Int32(nblk):
                elem_off = blk_idx * Int32(blk)
                r_src = cute.make_rmem_tensor(blk, Float32)
                s_blk = cute.make_tensor(
                    s_bf16.iterator + elem_off,
                    cute.make_layout(blk),
                )
                r_src.store(s_blk.load().to(Float32))
                e8m0_byte = Int32(127)
                inv = Float32(1.0)
                if cutlass.const_expr(not _MEGA_QUANT_SKIP_ARITH):
                    # Block amax via a single MAX-reduce over |x| (quant_sfd_row
                    # idiom) instead of a 32-iter serial fmax chain -> shorter
                    # dependency depth, far fewer issue slots (the dominant cost
                    # per ncu was short-scoreboard smem-dependency stalls, not the
                    # math itself).  scale = max(amax/448, 2^-30) -> e8m0; the
                    # floor avoids the zero-amax NaN (e8m0 has no zero).
                    acc = r_src.load()
                    abs_ir = _cute_math.absf(acc.ir_value())
                    abs_frg = type(acc)(abs_ir, acc.shape, acc.dtype)
                    absmax = abs_frg.reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
                    scale_f32 = cute.arch.fmax(
                        absmax * Float32(1.0 / 448.0),
                        Float32(2.0**-30),
                    )
                    # Round scale UP to a power of two; derive the e8m0 byte and
                    # the exact reciprocal by f32 exponent bit-manip (cute's e8m0
                    # .to(f32) decode returns 0 -> inv=inf, a known bug).  2^e has
                    # f32 exp field e+127 and e8m0 decodes byte b as 2^(b-127),
                    # so byte = exp field (+1 if mantissa!=0, which also keeps
                    # |x/scale| <= 448).
                    r_s = cute.make_rmem_tensor(1, Float32)
                    r_s[0] = scale_f32
                    r_sbits = cute.recast_tensor(r_s, cutlass.Int32)
                    exp_field = (r_sbits[0] >> Int32(23)) & Int32(0xFF)
                    mant = r_sbits[0] & Int32(0x7FFFFF)
                    e8m0_byte = exp_field
                    if mant != Int32(0):
                        e8m0_byte = exp_field + Int32(1)
                    r_inv = cute.make_rmem_tensor(1, Int32)
                    r_inv[0] = (Int32(254) - e8m0_byte) << Int32(23)
                    inv = cute.recast_tensor(r_inv, Float32)[0]
                # Vectorized scale + pack: each f32x4 group is scaled by inv and
                # packed to fp8 via cvt.rn.satfinite.e4m3x2.f32 (SATURATING -> no
                # clamp loop, never NaN), writing each Int32 word straight to
                # the smem fp8 block (quant_sfd_row idiom: indexed st.shared.u32
                # to a recast-i32 view; static index j, dynamic block via the
                # iterator offset -- the proven safe combination).
                s_q_blk_i32 = cute.make_tensor(
                    s_q_i32.iterator + blk_idx * Int32(nqw),
                    cute.make_layout(nqw),
                )
                for j in cutlass.range_constexpr(nqw):
                    f4 = cute.make_rmem_tensor(4, Float32)
                    f4[0] = r_src[j * 4 + 0] * inv
                    f4[1] = r_src[j * 4 + 1] * inv
                    f4[2] = r_src[j * 4 + 2] * inv
                    f4[3] = r_src[j * 4 + 3] * inv
                    s_q_blk_i32[j] = cutlass.Int32(
                        cvt_f32x4_to_f8x4_pack_i32(f4, cutlass.Float8E4M3FN)
                    )
                if cutlass.const_expr(not _MEGA_QUANT_NO_SF_STORE):
                    # Direct 1-byte global store of the e8m0 scale (distinct
                    # addresses, no neighbor clobber).
                    stg_b8_raw(
                        peer_sf_addr + Int64(sf_chunk_base + blk_idx),
                        e8m0_byte,
                    )

    @cute.jit
    def token_back_by_push(
        self,
        pull_buffer_ptr,
        pull_mbar_ptr,
        fc2_output_workspace,
        fc2_done_counter,
        token_src_metadata,
        combine_output,
        token_back_schedule_counter,
        peer_rank_ptr_mapper,
        phase_bit,
        stored_num_tokens_per_expert,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
        chunk_bytes: cutlass.Constexpr[int],
        combine_output_q=None,
        combine_sf_q=None,
        combine_global_q=None,
        fc2_output_sf_workspace=None,
        fc2_output_global_workspace=None,
        q_scratch_ptr=None,
        sf_scratch_ptr=None,
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

        num_experts_per_lane: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        num_global_warps: cutlass.Constexpr[int] = num_sms * self.num_dispatch_warps
        schedule_mode = self.token_back_schedule_mode
        atomic_batch = self.token_back_atomic_batch

        # static: stride by the global warp count.  atomic_counter: consume one
        # slot of the current batch, refilling via one grid-scoped
        # atomicAdd(atomic_batch) when exhausted so fast warps keep stealing
        # work.  cuTeDSL forbids closures over enclosing locals -> pass all in.
        def update_token_idx(
            token_idx,
            batch_remaining,
            lane_idx,
            schedule_counter,
            schedule_mode,
            atomic_batch,
            num_global_warps,
        ):
            if cutlass.const_expr(schedule_mode == "atomic_counter"):
                batch_remaining = batch_remaining - Int32(1)
                if batch_remaining == Int32(0):
                    base = Int32(0)
                    if lane_idx == Int32(0):
                        base = cute.arch.atomic_add(
                            schedule_counter,
                            Int32(atomic_batch),
                            sem="relaxed",
                            scope="gpu",
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
                token_idx,
                batch_remaining,
                lane_idx,
                token_back_schedule_counter,
                schedule_mode,
                atomic_batch,
                num_global_warps,
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
                expert_pool_block_offset = expert_pool_block_offset + prev_block_count

                current_expert_idx = current_expert_idx + Int32(1)
                if current_expert_idx < Int32(self.num_experts_per_rank):
                    expert_start_idx = expert_end_idx
                    valid_value = Int32(0)
                    for i in cutlass.range_constexpr(0, num_experts_per_lane, 1):
                        if (
                            current_expert_idx
                            == Int32(i * self.warp_threads) + lane_idx
                        ):
                            valid_value = stored_num_tokens_per_expert[i]
                    total_for_expert = cute.arch.shuffle_sync(
                        valid_value,
                        current_expert_idx % Int32(self.warp_threads),
                    )
                    expert_end_idx = expert_end_idx + total_for_expert

                    cluster_tile_cnt = (
                        total_for_expert + Int32(self.cluster_tile_tokens) - Int32(1)
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
                is_remote_token_back = src_rank != Int32(self.local_rank)

                local_token_addr = fc2_output_workspace.iterator.toint() + Int64(
                    pool_token_idx
                ) * Int64(fc2_token_bytes)
                smem_ptr_warp = pull_buffer_ptr + warp_idx * Int32(chunk_bytes)
                mbar_ptr_warp = pull_mbar_ptr + warp_idx

                if cutlass.const_expr(self.combine_is_quantized):
                    # Architecture A: the fc2 epilogue already quantized to fp8 +
                    # e8m0 in the LOCAL workspace (fc2_token_bytes now fp8-sized),
                    # so token-back is a PURE byte-copy -- no re-quant, no re-load.
                    # Push the fp8 data (num_chunks x chunk_bytes) to combine_output_q
                    # and the e8m0 scale plane (combine_sf_token_bytes) to combine_sf_q.
                    # All copies are TMA load(async)->store(async): the bf16 branch
                    # proves no generic->async fence is needed.
                    peer_q_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
                        combine_output_q.iterator,
                        src_rank,
                    )
                    peer_sf_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
                        combine_sf_q.iterator,
                        src_rank,
                    )
                    cell_idx = Int64(src_token * Int32(self.num_topk) + src_topk)
                    peer_token_ptr_q = peer_q_ptr + cell_idx * Int64(
                        self.combine_q_token_bytes
                    )
                    peer_token_ptr_sf = peer_sf_ptr + cell_idx * Int64(
                        self.combine_sf_token_bytes
                    )
                    local_sf_addr = fc2_output_sf_workspace.iterator.toint() + Int64(
                        pool_token_idx
                    ) * Int64(self.combine_sf_token_bytes)

                    if _iket_emit:
                        _iket.range_push("token_back_q")
                    cute.arch.sync_warp()

                    for chunk in cutlass.range(num_chunks, unroll=1):
                        chunk_off = Int64(chunk * chunk_bytes)
                        with cute.arch.elect_one():
                            tma_load_1d_raw(
                                smem_ptr_warp,
                                local_token_addr + chunk_off,
                                mbar_ptr_warp,
                                Int32(chunk_bytes),
                            )
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_ptr_warp,
                                Int32(chunk_bytes),
                            )
                            cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                            if cutlass.const_expr(not _MEGA_NO_COMBINE_STORE):
                                tma_store_1d(
                                    peer_token_ptr_q
                                    + Int64(chunk * self.combine_q_chunk_bytes),
                                    smem_ptr_warp,
                                    Int32(self.combine_q_chunk_bytes),
                                )
                        phase_bit = phase_bit ^ Int32(1)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)

                    # e8m0 scale plane: one TMA byte-copy of combine_sf_token_bytes
                    # (reuses the per-warp pull buffer; the data stores above have
                    # drained via wait_group(0) so the slot is free).
                    # nvfp4 packs its per-16 sfc into the combined global plane, so it
                    # never pushes a separate sfc plane; mxfp8 still pushes its e8m0 SF.
                    if cutlass.const_expr(
                        not _MEGA_NO_COMBINE_STORE and not self.combine_has_global
                    ):
                        with cute.arch.elect_one():
                            tma_load_1d_raw(
                                smem_ptr_warp,
                                local_sf_addr,
                                mbar_ptr_warp,
                                Int32(self.combine_sf_token_bytes),
                            )
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_ptr_warp,
                                Int32(self.combine_sf_token_bytes),
                            )
                            cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                            tma_store_1d(
                                peer_token_ptr_sf,
                                smem_ptr_warp,
                                Int32(self.combine_sf_token_bytes),
                            )
                        phase_bit = phase_bit ^ Int32(1)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)

                    # nvfp4 only: per-32 fp32 global plane -- a third TMA byte-copy
                    # of combine_global_token_bytes (same drained-pull-buffer idiom;
                    # 896 B for h7168 fits the per-warp pull slot, 16 B aligned).
                    if cutlass.const_expr(self.combine_has_global):
                        if cutlass.const_expr(not _MEGA_NO_COMBINE_STORE):
                            peer_global_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
                                combine_global_q.iterator,
                                src_rank,
                            )
                            peer_token_ptr_global = peer_global_ptr + cell_idx * Int64(
                                self.combine_global_token_bytes
                            )
                            local_global_addr = (
                                fc2_output_global_workspace.iterator.toint()
                                + Int64(pool_token_idx)
                                * Int64(self.combine_global_token_bytes)
                            )
                            with cute.arch.elect_one():
                                tma_load_1d_raw(
                                    smem_ptr_warp,
                                    local_global_addr,
                                    mbar_ptr_warp,
                                    Int32(self.combine_global_token_bytes),
                                )
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar_ptr_warp,
                                    Int32(self.combine_global_token_bytes),
                                )
                                cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                                tma_store_1d(
                                    peer_token_ptr_global,
                                    smem_ptr_warp,
                                    Int32(self.combine_global_token_bytes),
                                )
                            phase_bit = phase_bit ^ Int32(1)
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0)

                    if _iket_emit:
                        _iket.range_pop()
                else:
                    peer_combine_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
                        combine_output.iterator,
                        src_rank,
                    )
                    if cutlass.const_expr(self.token_back_reduce_topk):
                        peer_token_offset = Int64(src_token) * Int64(fc2_token_bytes)
                    else:
                        peer_token_offset = Int64(
                            src_token * Int32(self.num_topk) + src_topk
                        ) * Int64(fc2_token_bytes)
                    peer_token_ptr = peer_combine_ptr + peer_token_offset

                    if _iket_emit:
                        _iket.range_push("token_back")
                    cute.arch.sync_warp()

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
                                mbar_ptr_warp,
                                this_bytes,
                            )
                            cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                            if cutlass.const_expr(not _MEGA_NO_COMBINE_STORE):
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
                        if cutlass.const_expr(not _MEGA_NO_COMBINE_STORE):
                            if is_remote_token_back and remain_experts > Int32(4):
                                avg_token_back_window = self._adaptive_pace(
                                    avg_token_back_window,
                                    current_window,
                                    lo=1000,
                                    hi=5000,
                                )

                    if _iket_emit:
                        _iket.range_pop()

                token_idx, batch_remaining = update_token_idx(
                    token_idx,
                    batch_remaining,
                    lane_idx,
                    token_back_schedule_counter,
                    schedule_mode,
                    atomic_batch,
                    num_global_warps,
                )
        # if lane_idx == 0:
        #     cute.printf("<{}>", avg_token_back_window)

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
        slot: cutlass.Constexpr[int],
        num_sms,
        prologue_grid_sync: cutlass.Constexpr[bool],
        epilogue_grid_sync: cutlass.Constexpr[bool],
    ):
        # software_grid_sync expects a dispatch-group-relative thread id.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

        if prologue_grid_sync:
            software_grid_sync(
                grid_sync_counter,
                sm_idx,
                num_sms,
                tid_in_group,
                num_threads=self.num_dispatch_threads,
            )

        if sm_idx == 0:
            if warp_idx == 0:
                signal_phase = Int32(slot)
                signal_delta = Int32(1)
                target = Int32(self.world_size)
                if cutlass.const_expr(nvlink_barrier_counter is not None):
                    status = nvlink_barrier_counter[0] & Int32(3)
                    signal_phase = status & Int32(1)
                    signal_sign = status >> Int32(1)
                    if signal_sign != Int32(0):
                        signal_delta = Int32(-1)
                        target = Int32(0)

                nbs_local_base = nvlink_barrier_signal.iterator.toint()
                if lane_idx < Int32(self.world_size):
                    lane_peer_addr = peer_rank_ptr_mapper.map(
                        nbs_local_base,
                        lane_idx,
                        Int64(signal_phase * Int32(4)),
                    )
                    red_add_release_sys_s32_raw(lane_peer_addr, signal_delta)
                cute.arch.sync_warp()

                if lane_idx == 0:
                    if cutlass.const_expr(nvlink_barrier_counter is not None):
                        cute.arch.atomic_add(
                            nvlink_barrier_counter.iterator,
                            Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                    local_signal_ptr = nvlink_barrier_signal.iterator + signal_phase
                    if cutlass.const_expr(nvlink_barrier_counter is None):
                        while (
                            cute.arch.load(
                                local_signal_ptr, Int32, sem="acquire", scope="sys"
                            )
                            < target
                        ):
                            pass
                    else:
                        while (
                            cute.arch.load(
                                local_signal_ptr, Int32, sem="acquire", scope="sys"
                            )
                            != target
                        ):
                            pass

        if epilogue_grid_sync:
            software_grid_sync(
                grid_sync_counter,
                sm_idx,
                num_sms,
                tid_in_group,
                num_threads=self.num_dispatch_threads,
            )

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
            + Int32(self.cluster_shape_mn[1]) * Int32(bidy)
            + Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0]) * Int32(bidz)
        )
        local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)

        iket_active = (cta_linear_id == Int32(0)) and (local_warp_idx == Int32(0))
        if iket_active:
            _iket.range_push("Dispatch_Prep")

        self.dispatch_prep(
            token_comm_storage,
            token_comm_args.topk_idx,
            token_comm_args.expert_send_count,
            token_comm_args.src_token_topk_idx,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
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
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
        )

        if iket_active:
            _iket.range_pop()

        if cutlass.const_expr(
            self.enable_token_back and not self.token_back_standalone
        ):
            if iket_active:
                _iket.range_push("Token_Back_By_Push")

            if cutlass.const_expr(self.combine_is_quantized):
                # byte-copy stages through pull_buffer; no quant scratch.
                q_scratch_ptr = None
                sf_scratch_ptr = None
                combine_output_q = token_comm_args.combine_output_q
                combine_sf_q = token_comm_args.combine_sf_q
                combine_global_q = token_comm_args.combine_global_q
                fc2_output_global_workspace = (
                    token_comm_args.fc2_output_global_workspace
                )
            else:
                q_scratch_ptr = None
                sf_scratch_ptr = None
                combine_output_q = None
                combine_sf_q = None
                combine_global_q = None
                fc2_output_global_workspace = None

            self.token_back_by_push(
                token_comm_storage.pull_buffer.data_ptr(),
                token_comm_storage.pull_mbar.data_ptr(),
                token_comm_args.fc2_output_workspace,
                token_comm_args.fc2_done_counter,
                token_comm_args.token_src_metadata,
                token_comm_args.combine_output,
                token_comm_args.token_back_schedule_counter,
                token_comm_args.peer_rank_ptr_mapper,
                phase_bit,
                stored_num_tokens_per_expert,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                num_sms=token_comm_args.sm_count,
                chunk_bytes=self.hidden_bytes,
                combine_output_q=combine_output_q,
                combine_sf_q=combine_sf_q,
                combine_global_q=combine_global_q,
                fc2_output_sf_workspace=token_comm_args.fc2_output_sf_workspace,
                fc2_output_global_workspace=fc2_output_global_workspace,
                q_scratch_ptr=q_scratch_ptr,
                sf_scratch_ptr=sf_scratch_ptr,
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
            + Int32(self.cluster_shape_mn[1]) * Int32(bidy)
            + Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0]) * Int32(bidz)
        )
        local_warp_idx = Int32(warp_idx) - Int32(self.token_back_warp_start)
        if cutlass.const_expr(self.combine_is_quantized):
            # byte-copy needs combine_output_q / combine_sf_q /
            # fc2_output_sf_workspace, which this standalone call site does not
            # thread.  EP runs use reuse_dispatch_warps, so this is an assert.
            raise NotImplementedError(
                "quantized combine (mxfp8/nvfp4) is not wired for "
                "standalone_warps token-back; use reuse_dispatch_warps."
            )

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

        iket_active = (cta_linear_id == Int32(0)) and (local_warp_idx == Int32(0))
        if iket_active:
            _iket.range_push("Token_Back_By_Push_Standalone")

        self.token_back_by_push(
            tb_pull_buffer_ptr,
            tb_pull_mbar_ptr,
            token_comm_args.fc2_output_workspace,
            token_comm_args.fc2_done_counter,
            token_comm_args.token_src_metadata,
            token_comm_args.combine_output,
            token_comm_args.token_back_schedule_counter,
            token_comm_args.peer_rank_ptr_mapper,
            Int32(0),
            stored_num_tokens_per_expert,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
            chunk_bytes=self.tb_chunk_bytes,
        )

        if iket_active:
            _iket.range_pop()

    @cute.jit
    def tail_reset_shared_counters(
        self,
        token_comm_args,
        *,
        cta_linear_id,
        local_warp_idx,
        lane_idx,
    ):
        thread_linear = (
            cta_linear_id * Int32(self.num_dispatch_warps) + local_warp_idx
        ) * Int32(self.warp_threads) + lane_idx
        stride = Int32(token_comm_args.sm_count * self.num_dispatch_threads)

        recv_total: cutlass.Constexpr[int] = self.world_size * self.num_experts_per_rank
        i = thread_linear
        while i < Int32(recv_total):
            rank_idx = i // Int32(self.num_experts_per_rank)
            expert_idx = i % Int32(self.num_experts_per_rank)
            token_comm_args.expert_recv_count[rank_idx, expert_idx] = Int64(0)
            i = i + stride

        i = thread_linear
        while i < Int32(self.num_experts_per_rank):
            token_comm_args.expert_recv_count_sum[i] = Int64(0)
            i = i + stride

        if cutlass.const_expr(self.enable_token_back):
            i = thread_linear
            while i < Int32(self.num_experts_per_rank):
                token_comm_args.fc2_done_counter[i] = Int32(0)
                i = i + stride

        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            if thread_linear == Int32(0):
                token_comm_args.token_back_schedule_counter.store(Int32(0))

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
                + Int32(self.cluster_shape_mn[1]) * Int32(bidy)
                + Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0])
                * Int32(bidz)
            )
            local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                slot=1,
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
                slot=1,
                num_sms=token_comm_args.sm_count,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
            self.tail_reset_shared_counters(
                token_comm_args,
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
                slot=0,
                num_sms=token_comm_args.sm_count,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
