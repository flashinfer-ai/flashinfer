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
    Boolean,
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
    bf16x2_to_f32x2,
    cp_reduce_async_bulk_add_noftz_bf16_s2g,
    f32x2_to_bf16x2,
    fns_b32,
    ldg_b128_raw,
    ldg_b32_raw,
    ldg_f32_raw,
    read_clock64,
    stg_b128_raw,
    red_add_relaxed_sys_u64_raw,
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
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
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
    # Internode (NCCL GIN) fields; all None on the single-node path, so they
    # drop out of the kernel ABI there.
    "gin_dev_comm",
    "gin_window",
    "gin_recv_topk_idx",
    "gin_recv_topk_weights",
    "gin_recv_token",
    "gin_recv_sf",
    "gin_reduce_buf",
    "gin_remote_partial",
    "gin_payload_flag",
    "gin_token_count",
    "gin_epoch",
)

# ncclGinSignalOp_t (nccl_device/gin/gin_device_host_common.h): Inc = 0, Add = 1.
# The internode path only ever uses Add (Inc must not be mixed with Add on one
# signal without a reset).
_GIN_SIGNAL_OP_ADD = 1

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
        gin_dev_comm: Any = None,
        gin_window: Any = None,
        gin_recv_topk_idx: cute.Tensor = None,
        gin_recv_topk_weights: cute.Tensor = None,
        gin_recv_token: cute.Tensor = None,
        gin_recv_sf: cute.Tensor = None,
        gin_reduce_buf: cute.Tensor = None,
        gin_remote_partial: cute.Tensor = None,
        gin_payload_flag: cute.Tensor = None,
        gin_token_count: cute.Tensor = None,
        gin_epoch: cute.Tensor = None,
    ):
        self.gin_dev_comm = gin_dev_comm
        self.gin_window = gin_window
        self.gin_recv_topk_idx = gin_recv_topk_idx
        self.gin_recv_topk_weights = gin_recv_topk_weights
        self.gin_recv_token = gin_recv_token
        self.gin_recv_sf = gin_recv_sf
        self.gin_reduce_buf = gin_reduce_buf
        self.gin_remote_partial = gin_remote_partial
        self.gin_payload_flag = gin_payload_flag
        self.gin_token_count = gin_token_count
        self.gin_epoch = gin_epoch
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
        active_dispatch_warps: int = 4,
        compact_pull_buffer: bool = False,
        # Tail-split pair tasks: an expert with an odd CTA token-tile count publishes
        # fc2_done this many times for its last cluster tile (0 = off);
        # cta_tile_tokens (cluster_tile_tokens / 2) detects the odd count.
        fc2_publishes_per_split_tail_tile: int = 0,
        cta_tile_tokens: Optional[int] = None,
        # Internode (NCCL GIN, rail proxy pull; see _init_internode).
        # ranks_per_node=None means one NVLink domain spanning the world.
        ranks_per_node: Optional[int] = None,
        enable_nccl_gin: bool = False,
        global_rank: int = 0,
        max_tokens_per_rank: int = 0,
    ) -> None:
        self.world_size = world_size
        self._init_internode(
            ranks_per_node=ranks_per_node,
            enable_nccl_gin=enable_nccl_gin,
            global_rank=global_rank,
            max_tokens_per_rank=max_tokens_per_rank,
            num_experts_per_rank=num_experts_per_rank,
            num_topk=num_topk,
        )
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

        if token_back_schedule_mode not in ("static", "atomic_counter"):
            raise ValueError(
                "token_back_schedule_mode must be 'static' or "
                f"'atomic_counter'; got {token_back_schedule_mode!r}."
            )
        self.token_back_schedule_mode = token_back_schedule_mode
        # How many of the (warpgroup-aligned) dispatch warps do token-comm
        # work AT ALL (prep + barrier + pull + reuse token-back).  The
        # physical warp count stays num_dispatch_warps -- the setmaxnreg
        # register reallocation is warpgroup(4)-granular -- but the warps
        # beyond active_dispatch_warps skip the whole dispatch body and only
        # rejoin at kernel_tail, leaving them fully idle for future use.
        # Sizing: sustained pull bandwidth ~= active_warps * SMs *
        # hidden_bytes / read-RTT (one row in flight per warp); on H200
        # (450 GB/s unidirectional) even 1 warp/SM (~132 rows in flight)
        # exceeds the bandwidth-delay product, and measured EP4/EP8 sweeps
        # rank 1 > 2 > 4 warps -- the shallowest NVLink read queue wins
        # (a deep queue lengthens per-request RTT and fights the GEMM for
        # L2/issue slots).  Deeper per-warp pipelining (donating idle
        # warps' smem slots as extra in-flight stages) was measured at
        # -0.8%..+0.1% net -- in-flight depth is not the bottleneck.
        if active_dispatch_warps not in (1, 2, self.num_dispatch_warps):
            raise ValueError(
                "active_dispatch_warps must be 1, 2, or "
                f"{self.num_dispatch_warps}; got {active_dispatch_warps}."
            )
        self.active_dispatch_warps = active_dispatch_warps
        self.active_dispatch_threads = active_dispatch_warps * self.warp_threads
        # SMEM pull-buffer slots.  By default every physical dispatch warp
        # owns a hidden_bytes slot even when it is idle; with
        # compact_pull_buffer only the active warps get a slot (the idle warps
        # never enter dispatch_warp_body, and the reuse token-back walkers are
        # exactly the active warps and index the same slots with
        # tb_chunk_bytes strides).  The three idle slots cost 3 * hidden_bytes
        # (21 KiB at hidden=7168 fp8) of AB-stage SMEM per CTA.
        self.compact_pull_buffer = bool(compact_pull_buffer)
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
            self.active_dispatch_warps + 1 + self.num_token_back_warps
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
            if fc2_publishes_per_split_tail_tile < 0:
                raise ValueError(
                    "fc2_publishes_per_split_tail_tile must be >= 0, got "
                    f"{fc2_publishes_per_split_tail_tile}."
                )
            if fc2_publishes_per_split_tail_tile > 0:
                if (
                    cta_tile_tokens is None
                    or cta_tile_tokens * 2 != cluster_tile_tokens
                ):
                    raise ValueError(
                        "fc2_publishes_per_split_tail_tile > 0 requires "
                        "cta_tile_tokens * 2 == cluster_tile_tokens; got "
                        f"cta_tile_tokens={cta_tile_tokens}, "
                        f"cluster_tile_tokens={cluster_tile_tokens}."
                    )
                if (
                    fc2_publishes_per_split_tail_tile
                    > fc2_publishes_per_token_cluster_tile
                ):
                    raise ValueError(
                        "fc2_publishes_per_split_tail_tile "
                        f"({fc2_publishes_per_split_tail_tile}) cannot exceed "
                        "fc2_publishes_per_token_cluster_tile "
                        f"({fc2_publishes_per_token_cluster_tile})."
                    )
            self.fc2_publishes_per_split_tail_tile = (
                fc2_publishes_per_split_tail_tile
            )
            self.cta_tile_tokens = (
                cta_tile_tokens if fc2_publishes_per_split_tail_tile > 0 else None
            )
        else:
            self.fc2_token_bytes = 0
            self.fc2_num_chunks = 0
            self.fc2_publishes_per_token_cluster_tile = 0
            self.fc2_publishes_per_split_tail_tile = 0
            self.cta_tile_tokens = None

    # ------------------------------------------------------------------
    # Internode (NCCL GIN) configuration
    # ------------------------------------------------------------------

    def _init_internode(
        self,
        *,
        ranks_per_node,
        enable_nccl_gin,
        global_rank,
        max_tokens_per_rank,
        num_experts_per_rank,
        num_topk,
    ) -> None:
        """Derive the two-level EP constants.

        Ranks of one NVLink domain form a node (``ranks_per_node`` of them);
        every rank talks RDMA only to its rail peers (same local index on the
        other nodes).  Remote node ``d`` is addressed by its slot
        ``s = d if d < my_node else d - 1``.  On a single node every value
        degenerates to the original intra-node protocol.
        """
        world = self.world_size
        r = world if ranks_per_node is None else int(ranks_per_node)
        if r <= 0 or world % r != 0:
            raise ValueError(
                f"ranks_per_node={r} must be positive and divide world_size={world}."
            )
        self.ranks_per_node = r
        self.num_nodes = world // r
        self.enable_nccl_gin = bool(enable_nccl_gin)
        if self.num_nodes > 1 and not self.enable_nccl_gin:
            raise ValueError(
                "world_size spans several NVLink domains (ranks_per_node="
                f"{r}, world_size={world}); enable_nccl_gin=True is required."
            )
        if self.enable_nccl_gin and self.num_nodes < 2:
            raise ValueError("enable_nccl_gin requires at least two nodes.")
        self.global_rank = int(global_rank)
        self.my_node = self.global_rank // r
        self.local_rank_in_node = self.global_rank % r
        self.node_rank_base = self.my_node * r
        self.num_remote_slots = self.num_nodes - 1
        # Node-local experts form one contiguous global id range.
        self.node_expert_count = r * num_experts_per_rank
        self.node_expert_base = self.my_node * self.node_expert_count
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        if self.enable_nccl_gin:
            if world > 32:
                raise ValueError(
                    "the internode pull keeps one warp lane per source rank; "
                    f"world_size={world} > 32 is not supported."
                )
            if num_topk > 32:
                raise ValueError(f"num_topk={num_topk} > 32 is not supported.")
            if self.max_tokens_per_rank <= 0:
                raise ValueError("enable_nccl_gin requires max_tokens_per_rank > 0.")
        # GIN signal map: META (topk_idx landed), PAY (weights / SF / tokens
        # landed, three puts), RET (node partials returned).
        s = self.num_remote_slots
        self.gin_meta_signal_base = 0
        self.gin_pay_signal_base = s
        self.gin_ret_signal_base = 2 * s
        self.gin_signal_count = 3 * s
        self.gin_pay_puts = 3
        # Developer-only progress stamps (MEGA_GIN_DEBUG_ADDR = device-visible
        # pinned host address); baked in at trace time, off by default.
        self._dbg_addr = int(os.environ.get("MEGA_GIN_DEBUG_ADDR", "0"))

    @cute.jit
    def _dbg(self, cta, phase: cutlass.Constexpr[int], warp_idx, lane_idx):
        """Developer-only: record when CTA ``cta`` reached ``phase`` (%globaltimer, host-visible)."""
        if cutlass.const_expr(self._dbg_addr != 0):
            if warp_idx == Int32(0) and lane_idx == Int32(0):
                addr = Int64(self._dbg_addr) + (
                    Int64(cta) * Int64(64) + Int64(phase)
                ) * Int64(4)
                now = Int64(
                    llvm.inline_asm(
                        T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
                        has_side_effects=True, asm_dialect=0,
                    )
                )
                cute.arch.store(
                    cute.make_ptr(Int32, addr, AddressSpace.gmem, assumed_align=4),
                    Int32(now & Int64(0x7FFFFFFF)) | Int32(1),
                    sem="relaxed",
                    scope="sys",
                )

    @staticmethod
    def internode_signal_count(num_nodes: int) -> int:
        """Number of GIN signals the internode path needs (host-side sizing)."""
        return 3 * max(num_nodes - 1, 0)

    def prepare_device_resources(self, token_comm_args) -> None:
        """Materialize the GIN handles at kernel entry.

        ``DevComm`` is a by-value kernel argument whose nccl4py pointer is
        materialized lazily; it has to happen in a region that dominates every
        consumer, so the kernel calls this as its first statement and only the
        scalar GIN pointer and rail-team value are carried into the nested
        regions (passing the native wrappers through nested JIT calls breaks
        MLIR dominance).
        """
        if not self.enable_nccl_gin:
            return
        from nccl.core.device import cute as nccl_cute

        dev_comm = token_comm_args.gin_dev_comm
        gin = dev_comm.gin(
            nccl_cute.GinBackendMask.GDAKI,
            context_id=0,
            resource_sharing_mode=nccl_cute.GinResourceSharingMode.GPU,
        )
        self._gin_ptr = gin.ptr
        self._team_rail_value = dev_comm.team_rail.__extract_mlir_values__()[0]

    def _gin(self):
        from nccl.core.device.cute.gin import Gin

        return Gin(ptr=self._gin_ptr)

    @cute.jit
    def _remote_slot_node(self, slot):
        """Remote slot -> node index (skips this rank's own node)."""
        node = Int32(slot)
        if node >= Int32(self.my_node):
            node = node + Int32(1)
        return node

    @cute.jit
    def _slot_at_node(self, dst_node):
        """Slot under which ``dst_node`` sees this rank's node."""
        slot = Int32(self.my_node)
        if Int32(self.my_node) > dst_node:
            slot = slot - Int32(1)
        return slot

    @cute.jit
    def _gin_wait_signal(self, signal_id: cutlass.Constexpr[int], delta: cutlass.Constexpr[int]):
        """Single-thread wait for this launch's ``delta`` increments of a signal.

        Signals are never reset (they are symmetric NCCL state that kernel
        replay cannot restore); each waiter advances the NCCL-managed shadow by
        the exact per-launch increment and waits for the signal to reach it.
        """
        from nccl.core.device import cute as nccl_cute

        gin = self._gin()
        shadow = gin.signal_shadow_pointer(signal=signal_id)
        old = cute.arch.atomic_add(
            shadow, cutlass.Uint64(delta), sem="relaxed", scope="gpu",
        )
        least = Int64(old) + Int64(delta)
        gin.wait_signal(
            nccl_cute.thread(),
            signal=signal_id,
            least=least,
            bits=64,
            ord=nccl_cute.MemoryOrder.ACQUIRE,
        )

    @property
    def enable_token_back(self) -> bool:
        # token-back warps run if they push the DATA plane, the SF plane, or both.
        return self.push_data or self.push_sf

    def pull_buffer_bytes(self) -> int:
        """Bytes of the dispatch pull buffer in ``extra_smem_storage_class``."""
        if self.compact_pull_buffer:
            # Each active warp needs one token row for dispatch and one
            # tb_chunk_bytes piece for the reuse token-back path.
            return self.active_dispatch_warps * max(
                self.hidden_bytes, self.tb_chunk_bytes
            )
        return self.num_dispatch_warps * self.hidden_bytes

    def extra_smem_storage_class(self) -> type:
        pull_buffer_bytes = self.pull_buffer_bytes()
        num_total_experts = self.num_total_experts

        if self.token_back_standalone:
            @cute.struct
            class TokenCommStorage:
                pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
                smem_expert_count: cute.struct.MemRange[
                    Int32, num_total_experts
                ]
                pull_buffer: cute.struct.Align[cute.struct.MemRange[Uint8, pull_buffer_bytes], 16]
                tb_pull_mbar: cute.struct.MemRange[Int64, self.num_token_back_warps]
                tb_pull_buffer: cute.struct.Align[cute.struct.MemRange[Uint8, self.num_token_back_warps * self.tb_chunk_bytes], 16]

            return TokenCommStorage

        @cute.struct
        class TokenCommStorage:
            pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
            smem_expert_count: cute.struct.MemRange[
                Int32, num_total_experts
            ]
            pull_buffer: cute.struct.Align[cute.struct.MemRange[Uint8, pull_buffer_bytes], 16]

        return TokenCommStorage

    def fc1_ready_counter_ptr(self, token_comm_args):
        return token_comm_args.fc1_ready_counter.iterator

    @cute.jit
    def _cta_linear_id(self):
        """Grid-wide CTA id: ``bidz * cluster_size + %cluster_ctarank``.

        TokenComm receives the launch-view cluster shape, not the scheduler's
        internally swapped shape. The grid is (cluster_m, cluster_n, clusters)
        for both operand orders, so bidy's multiplier is cluster_m. Matching
        the hardware x-fastest cluster-rank enumeration gives a bijection onto
        [0, sm_count) -- required by the dispatch work partition and the
        tail bulk-zero striding.
        """
        bidx, bidy, bidz = cute.arch.block_idx()
        bidx_extent = self.cluster_shape_mn[0]
        return (
            Int32(bidx)
            + Int32(bidx_extent) * Int32(bidy)
            + Int32(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
            * Int32(bidz)
        )

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
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_tokens,
        num_sms,
        count_group: cutlass.Constexpr[int] = 0,
        token_count_ptr=None,
    ):
        """Histogram, reserve and publish the (token, topk) slots of one source.

        ``local_rank`` is the SOURCE id written into the owners'
        ``src_token_topk_idx[e][src][slot]`` / ``expert_recv_count[src][e]``:
        this rank's own (global) id, or -- on the internode path, when a rail
        peer replays a remote origin's ``topk_idx`` -- that origin's global id.
        ``count_group`` selects the reservation counter set in
        ``expert_send_count`` (one set of ``num_total_experts`` per source).
        Internode: only experts of this node are handled (owners on other
        nodes are reached through their own rail peer), and
        ``token_count_ptr`` receives ``max(valid token) + 1``.
        """
        node_filter: cutlass.Constexpr[bool] = self.enable_nccl_gin
        expert_lo: cutlass.Constexpr[int] = (
            self.node_expert_base if node_filter else 0
        )
        expert_cnt: cutlass.Constexpr[int] = (
            self.node_expert_count if node_filter else self.num_total_experts
        )
        count_base: cutlass.Constexpr[int] = count_group * self.num_total_experts
        thread_idx_in_dispatch = Int32(warp_idx * self.warp_threads + lane_idx)
        smem_count_ptr = token_comm_storage.smem_expert_count.data_ptr()
        i = thread_idx_in_dispatch
        while i < Int32(self.num_total_experts):
            (smem_count_ptr + i).store(Int32(0))
            i = i + Int32(self.active_dispatch_threads)
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.active_dispatch_threads,
        )

        tokens_per_warp: cutlass.Constexpr[int] = 32 // self.num_topk
        active_lanes: cutlass.Constexpr[int] = tokens_per_warp * self.num_topk
        num_dispatch_warps_per_grid: cutlass.Constexpr[int] = (
            num_sms * self.active_dispatch_warps
        )

        base_token_for_warp = (
            sm_idx * self.active_dispatch_warps + warp_idx
        ) * tokens_per_warp
        grid_token_stride = num_dispatch_warps_per_grid * tokens_per_warp

        max_valid_token_end = Int32(0)
        t = base_token_for_warp
        while t < num_tokens:
            token_offset_in_warp = lane_idx // self.num_topk
            token_global = t + token_offset_in_warp
            if lane_idx < active_lanes and token_global < num_tokens:
                topk_slot = lane_idx % self.num_topk
                expert_id = Int32(topk_idx[token_global, topk_slot])
                if expert_id >= Int32(0):
                    if cutlass.const_expr(token_count_ptr is not None):
                        max_valid_token_end = cutlass.max(
                            max_valid_token_end, token_global + Int32(1)
                        )
                    in_range = Boolean(True)
                    if cutlass.const_expr(node_filter):
                        in_range = (expert_id - Int32(expert_lo)) < Int32(
                            expert_cnt
                        ) and expert_id >= Int32(expert_lo)
                    if in_range:
                        cute.arch.atomic_add(
                            smem_count_ptr + expert_id,
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
            cute.arch.sync_warp()
            t += grid_token_stride

        if cutlass.const_expr(token_count_ptr is not None):
            warp_max = Int32(
                cute.arch.warp_redux_sync(max_valid_token_end, "max")
            )
            if lane_idx == Int32(0) and warp_max > Int32(0):
                cute.arch.atomic_max(
                    token_count_ptr, warp_max, sem="relaxed", scope="gpu",
                )

        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.active_dispatch_threads,
        )

        for offset in cutlass.range_constexpr(
            0, expert_cnt, self.active_dispatch_threads,
        ):
            expert_id = Int32(
                expert_lo + offset + warp_idx * self.warp_threads + lane_idx
            )
            if expert_id < Int32(expert_lo + expert_cnt):
                slot_ptr = smem_count_ptr + expert_id
                local_count = (slot_ptr).load()
                delta = (Int64(1) << Int64(32)) | (Int64(local_count) & Int64(0xFFFFFFFF))
                old_packed = cute.arch.atomic_add(
                    expert_send_count.iterator + (expert_id + Int32(count_base)),
                    delta,
                    sem="relaxed",
                    scope="gpu",
                )
                base_slot = Int32(old_packed & Int64(0xFFFFFFFF))
                (slot_ptr).store(base_slot)
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.active_dispatch_threads,
        )

        t = base_token_for_warp
        while t < num_tokens:
            token_offset_in_warp = lane_idx // self.num_topk
            token_global = t + token_offset_in_warp
            if lane_idx < active_lanes and token_global < num_tokens:
                topk_slot = lane_idx % self.num_topk
                expert_id = Int32(topk_idx[token_global, topk_slot])
                in_range = expert_id >= Int32(0)
                if cutlass.const_expr(node_filter):
                    in_range = in_range and (
                        (expert_id - Int32(expert_lo)) < Int32(expert_cnt)
                    ) and expert_id >= Int32(expert_lo)
                if in_range:
                    # Owner index in the peer mapper: node-local on the
                    # internode path (the mapper only covers this node).
                    dst_rank = expert_id // Int32(self.num_experts_per_rank)
                    if cutlass.const_expr(node_filter):
                        dst_rank = dst_rank - Int32(self.node_rank_base)
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

    # ------------------------------------------------------------------
    # Internode dispatch (rail proxy): Stage 1 puts, count exchange, barrier
    # ------------------------------------------------------------------

    @cute.jit
    def _gin_bytes(self, token_comm_args, addr, nbytes):
        """Byte view of an arena address as a ``gin.put`` operand.

        ``gin.put`` derives the window offset as ``iterator - win.local_pointer(0)``,
        and the window's local pointer is NCCL's LSA alias of this rank's
        buffer, not the allocator address the kernel otherwise uses.  The
        node peer table maps an arena address to every node rank's LSA alias,
        so mapping it to this rank's own slot yields the address the window
        offset is measured from.
        """
        alias = token_comm_args.peer_rank_ptr_mapper.map(
            addr, Int32(self.local_rank_in_node), Int64(0)
        )
        return cute.make_tensor(
            cute.make_ptr(Uint8, alias, AddressSpace.gmem, assumed_align=16),
            cute.make_layout(nbytes),
        )

    @cute.jit
    def _gin_stage1_put(self, token_comm_args, slot, kind, n_tokens):
        """Stage 1: one of the four puts that ship this rank's routing and tokens.

        Zero-copy: the puts read the symmetric input buffers directly.  One
        lane per (remote node ``slot``, ``kind``) posts its put, so the four
        WQE posts (about 10 us each) overlap instead of queueing on one lane.

        * kind 0, META: the full ``topk_idx`` (pad rows carry -1, so the
          receiver needs no token count), +1 on the META signal;
        * kind 1 / 2 / 3, PAY: ``topk_weights`` / scale words / token rows of
          the ``n_tokens`` valid rows, +1 each on the PAY signal.  Without a
          valid token, kind 1 raises PAY by 3 instead, so the receiver's
          per-launch increment stays constant.
        """
        from nccl.core.device import cute as nccl_cute

        args = token_comm_args
        gin = self._gin()
        win = args.gin_window
        team = self._team_rail_value
        coop = nccl_cute.thread()
        dst_node = self._remote_slot_node(slot)
        my_slot64 = Int64(self._slot_at_node(dst_node))
        t_max: cutlass.Constexpr[int] = self.max_tokens_per_rank
        meta_bytes: cutlass.Constexpr[int] = t_max * self.num_topk * 8
        w_row_bytes: cutlass.Constexpr[int] = self.num_topk * 4
        sf_row_bytes: cutlass.Constexpr[int] = self.sf_uint32_per_token * 4
        tok_row_bytes: cutlass.Constexpr[int] = self.hidden_bytes

        src_addr = args.topk_idx.iterator.toint()
        dst_addr = args.gin_recv_topk_idx.iterator.toint() + my_slot64 * Int64(meta_bytes)
        nbytes = Int32(meta_bytes)
        signal_id = Int32(self.gin_meta_signal_base) + Int32(my_slot64)
        if kind == Int32(1):
            src_addr = args.input_topk_weights_buffer.iterator.toint()
            dst_addr = args.gin_recv_topk_weights.iterator.toint() + my_slot64 * Int64(
                t_max * w_row_bytes
            )
            nbytes = n_tokens * Int32(w_row_bytes)
        if kind == Int32(2):
            src_addr = args.input_sf_buffer.iterator.toint()
            dst_addr = args.gin_recv_sf.iterator.toint() + my_slot64 * Int64(
                t_max * sf_row_bytes
            )
            nbytes = n_tokens * Int32(sf_row_bytes)
        if kind == Int32(3):
            src_addr = args.input_token_buffer.iterator.toint()
            dst_addr = args.gin_recv_token.iterator.toint() + my_slot64 * Int64(
                t_max * tok_row_bytes
            )
            nbytes = n_tokens * Int32(tok_row_bytes)
        if kind != Int32(0):
            signal_id = Int32(self.gin_pay_signal_base) + Int32(my_slot64)

        if nbytes > Int32(0):
            gin.put(
                team, dst_node, win,
                self._gin_bytes(args, dst_addr, nbytes),
                win,
                self._gin_bytes(args, src_addr, nbytes),
                coop,
                is_signal=True, signal_id=signal_id,
                signal_op=_GIN_SIGNAL_OP_ADD, signal_op_arg=Int64(1),
            )
        elif kind == Int32(1):
            gin.signal(
                team, dst_node, True, signal_id, _GIN_SIGNAL_OP_ADD,
                Int64(self.gin_pay_puts), coop,
            )

    @cute.jit
    def _exchange_counts(
        self,
        expert_send_count,
        expert_recv_count,
        expert_recv_count_sum,
        peer_rank_ptr_mapper,
        warp_idx,
        lane_idx,
        *,
        count_group: cutlass.Constexpr[int],
        src_index,
    ):
        """Publish one source's per-expert counts to this node's expert owners."""
        for offset in cutlass.range_constexpr(
            0, self.node_expert_count, self.active_dispatch_threads,
        ):
            e_rel = Int32(offset + warp_idx * self.warp_threads + lane_idx)
            if e_rel < Int32(self.node_expert_count):
                expert_id = e_rel + Int32(self.node_expert_base)
                dst_rank = (
                    expert_id // Int32(self.num_experts_per_rank)
                    - Int32(self.node_rank_base)
                )
                dst_local_expert = expert_id % Int32(self.num_experts_per_rank)
                status_u64 = cute.arch.load(
                    expert_send_count.iterator
                    + (expert_id + Int32(count_group * self.num_total_experts)),
                    Int64,
                    sem="relaxed",
                    scope="gpu",
                )
                token_count_u32 = Int32(status_u64 & Int64(0xFFFFFFFF))
                erc_peer_addr = peer_rank_ptr_mapper.map(
                    expert_recv_count.iterator.toint(),
                    dst_rank,
                    Int64(
                        (Int32(src_index) * Int32(self.num_experts_per_rank)
                         + dst_local_expert) * Int32(8)
                    ),
                )
                stg_b64_raw(erc_peer_addr, Int64(token_count_u32))
                ercs_peer_addr = peer_rank_ptr_mapper.map(
                    expert_recv_count_sum.iterator.toint(),
                    dst_rank,
                    Int64(dst_local_expert * Int32(8)),
                )
                red_add_relaxed_sys_u64_raw(ercs_peer_addr, status_u64)

    @cute.jit
    def dispatch_barrier_gin(
        self,
        token_comm_storage,
        token_comm_args,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
    ):
        """Internode dispatch barrier (design D2-D10).

        Own-token prep is already done.  CTA0 posts Stage 1 to every rail peer
        and publishes this rank's own counts; every CTA then replays the
        received remote ``topk_idx`` as proxy prep (source id = the remote
        origin's global rank), CTA0 publishes those counts, and the node
        NVLink barrier makes all counts visible before the scheduler starts.
        """
        args = token_comm_args
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx
        num_slots: cutlass.Constexpr[int] = self.num_remote_slots
        t_max: cutlass.Constexpr[int] = self.max_tokens_per_rank

        self._dbg(sm_idx, 1, warp_idx, lane_idx)
        # D2: own prep (slots, counts, token count) is grid-visible.
        software_grid_sync(args.grid_sync_counter, sm_idx, num_sms, tid_in_group,
                           num_threads=self.active_dispatch_threads)
        self._dbg(sm_idx, 2, warp_idx, lane_idx)

        if sm_idx == 0:
            if warp_idx == 0:
                # D3: Stage 1, one lane per remote node.
                n_own = Int32(
                    cute.arch.load(
                        args.gin_token_count.iterator, Int32,
                        sem="relaxed", scope="gpu",
                    )
                )
                # Lane 4*s + kind posts put ``kind`` to remote slot ``s``.
                if lane_idx < Int32(4 * num_slots):
                    self._gin_stage1_put(
                        args, lane_idx // Int32(4), lane_idx % Int32(4), n_own,
                    )
                cute.arch.sync_warp()
                self._dbg(sm_idx, 3, warp_idx, lane_idx)
            # D4: own counts to this node's owners.
            self._exchange_counts(
                args.expert_send_count,
                args.expert_recv_count,
                args.expert_recv_count_sum,
                args.peer_rank_ptr_mapper,
                warp_idx,
                lane_idx,
                count_group=0,
                src_index=Int32(self.global_rank),
            )
            cute.arch.fence_acq_rel_sys()
            # D5: wait for every rail peer's routing (META).
            if warp_idx == 0:
                if lane_idx == 0:
                    for s in cutlass.range_constexpr(num_slots):
                        self._gin_wait_signal(self.gin_meta_signal_base + s, 1)
                cute.arch.sync_warp()
                self._dbg(sm_idx, 4, warp_idx, lane_idx)

        # D6: the META acquire on CTA0 is published grid-wide.
        software_grid_sync(args.grid_sync_counter, sm_idx, num_sms, tid_in_group,
                           num_threads=self.active_dispatch_threads)
        self._dbg(sm_idx, 5, warp_idx, lane_idx)

        # D7: proxy prep of every remote origin whose rail peer is this rank.
        for s in cutlass.range_constexpr(num_slots):
            src_node: cutlass.Constexpr[int] = s if s < self.my_node else s + 1
            src_global_rank: cutlass.Constexpr[int] = (
                src_node * self.ranks_per_node + self.local_rank_in_node
            )
            recv_topk = cute.make_tensor(
                args.gin_recv_topk_idx.iterator + s * t_max * self.num_topk,
                cute.make_layout(
                    (t_max, self.num_topk), stride=(self.num_topk, 1)
                ),
            )
            self.dispatch_prep(
                token_comm_storage,
                recv_topk,
                args.expert_send_count,
                args.src_token_topk_idx,
                args.peer_rank_ptr_mapper,
                sm_idx,
                warp_idx,
                lane_idx,
                local_rank=Int32(src_global_rank),
                num_tokens=t_max,
                num_sms=num_sms,
                count_group=1 + s,
                token_count_ptr=args.gin_token_count.iterator + (1 + s),
            )

        self._dbg(sm_idx, 6, warp_idx, lane_idx)
        # D8: proxy reservations are grid-visible.
        software_grid_sync(args.grid_sync_counter, sm_idx, num_sms, tid_in_group,
                           num_threads=self.active_dispatch_threads)
        self._dbg(sm_idx, 7, warp_idx, lane_idx)

        # D9: proxy counts to this node's owners.
        if sm_idx == 0:
            for s in cutlass.range_constexpr(num_slots):
                src_node: cutlass.Constexpr[int] = s if s < self.my_node else s + 1
                self._exchange_counts(
                    args.expert_send_count,
                    args.expert_recv_count,
                    args.expert_recv_count_sum,
                    args.peer_rank_ptr_mapper,
                    warp_idx,
                    lane_idx,
                    count_group=1 + s,
                    src_index=Int32(
                        src_node * self.ranks_per_node + self.local_rank_in_node
                    ),
                )
            cute.arch.fence_acq_rel_sys()
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.active_dispatch_threads,
        )

        # D10: node barrier publishes every count before the scheduler starts.
        self.nvlink_barrier(
            args.nvlink_barrier_signal,
            args.nvlink_barrier_counter,
            args.grid_sync_counter,
            args.peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            num_sms=num_sms,
            prologue_grid_sync=False,
            epilogue_grid_sync=True,
            sync_threads=self.active_dispatch_threads,
        )
        self._dbg(sm_idx, 8, warp_idx, lane_idx)

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
                           num_threads=self.active_dispatch_threads)

        if sm_idx == 0:
            for offset in cutlass.range_constexpr(
                0, self.num_total_experts, self.active_dispatch_threads,
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
            number_of_threads=self.active_dispatch_threads,
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
            sync_threads=self.active_dispatch_threads,
        )
    @cute.jit
    def _gin_payload_gate(self, token_comm_args, sm_idx, warp_idx, lane_idx):
        """Before pulling remote-origin rows: payload of every rail peer landed.

        CTA0 of each rank waits for its own PAY signals (Stage 1 token /
        scale / weight puts) and then bumps its ``gin_payload_flag[s]`` with a
        system-scope release; every pulling warp acquires the flags of all
        node peers (each is the rail peer of a different remote origin set).
        Flags are monotonic launch counters (zeroed once at allocation), so a
        launch waits for ``epoch + 1``.
        """
        args = token_comm_args
        num_slots: cutlass.Constexpr[int] = self.num_remote_slots
        target = Int32(
            cute.arch.load(args.gin_epoch.iterator, Int32, sem="relaxed", scope="gpu")
        ) + Int32(1)
        if sm_idx == Int32(0) and warp_idx == Int32(0):
            if lane_idx == Int32(0):
                for s in cutlass.range_constexpr(num_slots):
                    self._gin_wait_signal(
                        self.gin_pay_signal_base + s, self.gin_pay_puts
                    )
                for s in cutlass.range_constexpr(num_slots):
                    red_add_release_sys_s32_raw(
                        (args.gin_payload_flag.iterator + s).toint(), Int32(1)
                    )
            cute.arch.sync_warp()
            self._dbg(sm_idx, 10, warp_idx, lane_idx)
        if lane_idx == Int32(0):
            flag_base = args.gin_payload_flag.iterator.toint()
            for peer in cutlass.range_constexpr(self.ranks_per_node):
                for s in cutlass.range_constexpr(num_slots):
                    flag_addr = args.peer_rank_ptr_mapper.map(
                        flag_base, Int32(peer), Int64(s * 4)
                    )
                    flag_ptr = cute.make_ptr(
                        Int32, flag_addr, AddressSpace.gmem, assumed_align=4
                    )
                    while cute.arch.load(
                        flag_ptr, Int32, sem="acquire", scope="sys"
                    ) < target:
                        _nanosleep(Int32(200))
            # The landed rows are read by TMA (async proxy) and by LDG.
            cute.arch.fence_proxy("async.global")
        cute.arch.sync_warp()
        self._dbg(sm_idx, 11, warp_idx, lane_idx)

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
        gin_args=None,
    ):
        """Pull every routed (token, topk) row into the FC1 pool.

        Pool rows of an expert are assigned by a rank-interleaved round robin
        over the source ranks (one warp lane per source).  Internode: sources
        are all ``world_size`` global ranks; this node's sources fill the
        first ``I_e`` rows of the expert and remote origins the rest, and the
        pull runs in two passes (node sources, then remote ones after the
        payload gate) so intra tiles never wait for RDMA.  A remote origin's
        row is read from its rail peer's Stage 1 landing buffers, and its
        metadata points the FC2 result at that rail peer's landing row
        ``T + slot * T + token`` of the combine plane.
        """
        gin: cutlass.Constexpr[bool] = self.enable_nccl_gin
        # MemRange does not support dynamic indexing here; use raw pointers.
        pull_mbar_ptr = token_comm_storage.pull_mbar.data_ptr()
        pull_buffer_ptr = token_comm_storage.pull_buffer.data_ptr()
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(pull_mbar_ptr + warp_idx, 1)
        cute.arch.sync_warp()

        phase_bit = Int32(0)

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

        num_global_warps: cutlass.Constexpr[int] = (
            num_sms * self.active_dispatch_warps
        )

        _iket_pull_emit = (
            (sm_idx == Int32(0))
            and (warp_idx == Int32(0))
            and (lane_idx == Int32(0))
        )

        # Source-lane classification (internode): lane = global source rank.
        lane_is_node_source = Boolean(True)
        lane_is_remote_source = Boolean(False)
        if cutlass.const_expr(gin):
            lane_node = lane_idx // Int32(self.ranks_per_node)
            lane_is_node_source = lane_node == Int32(self.my_node)
            lane_is_remote_source = lane_node != Int32(self.my_node)

        num_passes: cutlass.Constexpr[int] = 2 if gin else 1
        for pull_pass in cutlass.range_constexpr(num_passes):
            if cutlass.const_expr(gin and pull_pass == 1):
                self._dbg(sm_idx, 9, warp_idx, lane_idx)
                self._gin_payload_gate(gin_args, sm_idx, warp_idx, lane_idx)

            current_expert_idx = Int32(-1)
            expert_start_idx = Int32(0)
            expert_end_idx = Int32(0)
            expert_pool_block_offset = Int32(0)
            expert_task_tile_offset = Int32(0)
            # SF rows use their own padding; token and SF pool offsets can diverge.
            expert_sf_pool_block_offset = Int32(0)
            stored_rank_count_lane = Int32(0)
            node_source_total = Int32(0)
            token_idx = sm_idx * Int32(self.active_dispatch_warps) + warp_idx

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
                        if cutlass.const_expr(gin):
                            node_lane_count = Int32(0)
                            if lane_is_node_source:
                                node_lane_count = stored_rank_count_lane
                            node_source_total = Int32(
                                cute.arch.warp_redux_sync(node_lane_count, "add")
                            )

                    token_idx_in_expert = token_idx - expert_start_idx
                    slot_idx = token_idx_in_expert
                    offset = Int32(0)
                    remaining_lane = stored_rank_count_lane
                    do_row = Boolean(True)
                    if cutlass.const_expr(gin):
                        # Node sources own rows [0, I_e) of the expert, remote
                        # origins rows [I_e, total); each phase round-robins
                        # over its own source lanes only.
                        in_remote_rows = token_idx_in_expert >= node_source_total
                        if cutlass.const_expr(pull_pass == 0):
                            do_row = token_idx_in_expert < node_source_total
                        else:
                            do_row = token_idx_in_expert >= node_source_total
                        if in_remote_rows:
                            slot_idx = token_idx_in_expert - node_source_total
                            if lane_is_node_source:
                                remaining_lane = Int32(0)
                        else:
                            if lane_is_remote_source:
                                remaining_lane = Int32(0)

                    if do_row:
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
                        src_token = Int32(src_token_topk // Uint32(self.num_topk))
                        src_topk = Int32(src_token_topk % Uint32(self.num_topk))

                        inp_tok_local_base = input_token_buffer.iterator.toint()
                        # SF-free callers pass input_sf_buffer=None (their SF loops are
                        # compiled out by sf_uint32_per_token=0, so this base address
                        # is never consumed).
                        inp_sf_local_base = Int64(0)
                        if cutlass.const_expr(input_sf_buffer is not None):
                            inp_sf_local_base = input_sf_buffer.iterator.toint()
                        inp_w_local_base = input_topk_weights_buffer.iterator.toint()
                        # Peer index for the mapper and the token-back target.
                        src_peer = current_rank_in_expert_idx
                        md_src_token = src_token
                        if cutlass.const_expr(gin):
                            t_max: cutlass.Constexpr[int] = self.max_tokens_per_rank
                            src_node = current_rank_in_expert_idx // Int32(
                                self.ranks_per_node
                            )
                            src_peer = current_rank_in_expert_idx - src_node * Int32(
                                self.ranks_per_node
                            )
                            if src_node != Int32(self.my_node):
                                # Remote origin: read its rail peer's Stage 1
                                # landing buffers (same local index on this node).
                                src_slot = src_node
                                if src_node > Int32(self.my_node):
                                    src_slot = src_slot - Int32(1)
                                slot64 = Int64(src_slot)
                                inp_tok_local_base = (
                                    gin_args.gin_recv_token.iterator.toint()
                                    + slot64 * Int64(t_max * self.hidden_bytes)
                                )
                                inp_sf_local_base = (
                                    gin_args.gin_recv_sf.iterator.toint()
                                    + slot64 * Int64(t_max * self.sf_uint32_per_token * 4)
                                )
                                inp_w_local_base = (
                                    gin_args.gin_recv_topk_weights.iterator.toint()
                                    + slot64 * Int64(t_max * self.num_topk * 4)
                                )
                                md_src_token = (
                                    Int32(t_max) + src_slot * Int32(t_max) + src_token
                                )
                        cur_peer_offset = peer_rank_ptr_mapper.map(
                            Int64(0), src_peer, Int64(0)
                        )

                        with cute.arch.elect_one():
                            pull_buffer_warp_ptr = pull_buffer_ptr + (
                                warp_idx * Int32(self.hidden_bytes)
                            )
                            tma_src_addr = (
                                inp_tok_local_base
                                + cur_peer_offset
                                + Int64(src_token) * Int64(self.hidden_bytes)
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
                                src_rank=src_peer,
                                src_token=md_src_token,
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
                            Int32(0), self._flag_batch, task_tile_addr,
                        )
                        cute.arch.sync_warp()

                        if _iket_pull_emit:
                            _iket.range_pop()  # Pull.Arrival_Atomic

                        phase_bit = phase_bit ^ Int32(1)
                    else:
                        if _iket_pull_emit:
                            _iket.range_pop()  # Pull.ChooseToken

                    token_idx = token_idx + Int32(num_global_warps)
                else:
                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.ChooseToken

        # Tail flush: publish any leftover (< self._flag_batch) accumulated release.
        flag_tracker.fire()
        cute.arch.sync_warp()
        self._dbg(sm_idx, 12, warp_idx, lane_idx)

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
        num_token_back_warps: cutlass.Constexpr[int],
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
        num_global_warps: cutlass.Constexpr[int] = num_sms * num_token_back_warps
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
            token_idx = sm_idx * Int32(num_token_back_warps) + warp_idx
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
                    if cutlass.const_expr(
                        self.fc2_publishes_per_split_tail_tile > 0
                    ):
                        # Tail-split pair tasks: the last cluster tile of an expert with an odd CTA
                        # tile count publishes fc2_publishes_per_split_tail_tile times.
                        cta_tile_cnt = (
                            total_for_expert
                            + Int32(self.cta_tile_tokens - 1)
                        ) // Int32(self.cta_tile_tokens)
                        if (cta_tile_cnt & Int32(1)) == Int32(1):
                            cur_expert_expected = (
                                cur_expert_expected
                                - Int32(self.fc2_publishes_per_token_cluster_tile)
                                + Int32(self.fc2_publishes_per_split_tail_tile)
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

                # DATA plane: only the dispatch DATA path pushes here; epi_warps
                # has the epilogue STG/UBLK the data straight to the peer.
                if cutlass.const_expr(self.push_data):
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

                if cutlass.const_expr(self.push_sf):
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
        sync_threads: cutlass.Constexpr[int] = 0,
    ):
        # software_grid_sync expects a dispatch-group-relative thread id.
        # sync_threads: per-CTA arrival count for the grid syncs (0 -> the
        # full dispatch group; dispatch_barrier passes the ACTIVE subset).
        arrivals: cutlass.Constexpr[int] = (
            sync_threads if sync_threads > 0 else self.num_dispatch_threads
        )
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

        if prologue_grid_sync:
            software_grid_sync(grid_sync_counter, sm_idx, num_sms, tid_in_group,
                               num_threads=arrivals)

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
                # Node-scoped: the peer mapper only covers this NVLink domain
                # (ranks_per_node == world_size on a single node).
                signal_delta = Int32(1)
                target = Int32(self.ranks_per_node)
                if signal_sign != Int32(0):
                    signal_delta = Int32(-1)
                    target = Int32(0)

                nbs_local_base = nvlink_barrier_signal.iterator.toint()
                if lane_idx < Int32(self.ranks_per_node):
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
                               num_threads=arrivals)

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
        cta_linear_id = self._cta_linear_id()
        local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)

        # Record all four dispatch warps in CTA 0. Recording every persistent
        # CTA duplicates the same role and makes PIC-C's trace buffer too large.
        iket_active = cta_linear_id == Int32(0)
        if iket_active:
            _iket.range_push("Dispatch_Prep")

        if cutlass.const_expr(self.enable_nccl_gin):
            # Source id = this rank's GLOBAL id; owners are node-local.
            self.dispatch_prep(
                token_comm_storage,
                token_comm_args.topk_idx,
                token_comm_args.expert_send_count,
                token_comm_args.src_token_topk_idx,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                local_rank=Int32(self.global_rank),
                num_tokens=token_comm_args.input_token_buffer.shape[0],
                num_sms=token_comm_args.sm_count,
                count_group=0,
                token_count_ptr=token_comm_args.gin_token_count.iterator,
            )
        else:
            self.dispatch_prep(
                token_comm_storage,
                token_comm_args.topk_idx,
                token_comm_args.expert_send_count,
                token_comm_args.src_token_topk_idx,
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

        if cutlass.const_expr(self.enable_nccl_gin):
            self.dispatch_barrier_gin(
                token_comm_storage,
                token_comm_args,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                num_sms=token_comm_args.sm_count,
            )
        else:
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
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
            gin_args=(token_comm_args if self.enable_nccl_gin else None),
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
                num_token_back_warps=self.active_dispatch_warps,
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
        cta_linear_id = self._cta_linear_id()
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
            num_token_back_warps=self.num_token_back_warps,
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
    def _gin_node_reduce(self, token_comm_args, cta_linear_id, local_warp_idx, lane_idx):
        """Rail-peer node reduce (design C3).

        For every remote origin this rank proxies (slot ``s``) and every valid
        token ``t`` that hits this node, sum the landing rows
        ``combine[T + s*T + t][k]`` of the hitting top-k slots in ascending ``k``
        in FP32, round to BF16 once and store the node partial in
        ``gin_reduce_buf[s][t]``.  Grid-strided over all dispatch-group warps.
        """
        args = token_comm_args
        num_slots: cutlass.Constexpr[int] = self.num_remote_slots
        t_max: cutlass.Constexpr[int] = self.max_tokens_per_rank
        hidden: cutlass.Constexpr[int] = self.hidden
        num_topk: cutlass.Constexpr[int] = self.num_topk
        chunks_per_row: cutlass.Constexpr[int] = hidden // 8
        row_bytes: cutlass.Constexpr[int] = hidden * 2
        global_warp = cta_linear_id * Int32(self.num_dispatch_warps) + local_warp_idx
        num_global_warps = Int32(args.sm_count * self.num_dispatch_warps)
        combine_base = args.combine_output.iterator.toint()
        reduce_base = args.gin_reduce_buf.iterator.toint()
        for s in cutlass.range_constexpr(num_slots):
            n_s = Int32(
                cute.arch.load(
                    args.gin_token_count.iterator + (1 + s), Int32,
                    sem="relaxed", scope="gpu",
                )
            )
            t = global_warp
            while t < n_s:
                hit = Boolean(False)
                if lane_idx < Int32(num_topk):
                    e = Int32(args.gin_recv_topk_idx[s, t, lane_idx])
                    if e >= Int32(0):
                        hit = (e // Int32(self.node_expert_count)) == Int32(self.my_node)
                mask = Int32(cute.arch.vote_ballot_sync(hit))
                if mask != Int32(0):
                    landing_row = Int64(t_max + s * t_max) + Int64(t)
                    cell_base = combine_base + landing_row * Int64(num_topk * row_bytes)
                    out_base = reduce_base + (Int64(s * t_max) + Int64(t)) * Int64(row_bytes)
                    c = lane_idx
                    while c < Int32(chunks_per_row):
                        # Issue every slot's 16-byte load up front (independent
                        # addresses -> overlapping latencies); cells of slots
                        # that miss this node are stale but harmless to read
                        # and are never accumulated.
                        words = []
                        for k in cutlass.range_constexpr(num_topk):
                            words.append(
                                ldg_b128_raw(
                                    cell_base
                                    + Int64(k * row_bytes)
                                    + Int64(c) * Int64(16)
                                )
                            )
                        acc = [Float32(0.0) for _ in range(8)]
                        for k in cutlass.range_constexpr(num_topk):
                            if ((mask >> Int32(k)) & Int32(1)) != Int32(0):
                                for q in cutlass.range_constexpr(4):
                                    lo, hi = bf16x2_to_f32x2(words[k][q])
                                    acc[2 * q] = acc[2 * q] + lo
                                    acc[2 * q + 1] = acc[2 * q + 1] + hi
                        stg_b128_raw(
                            out_base + Int64(c) * Int64(16),
                            f32x2_to_bf16x2(acc[0], acc[1]),
                            f32x2_to_bf16x2(acc[2], acc[3]),
                            f32x2_to_bf16x2(acc[4], acc[5]),
                            f32x2_to_bf16x2(acc[6], acc[7]),
                        )
                        c = c + Int32(self.warp_threads)
                t = t + num_global_warps
        # Partials are read by the local NIC (the return put source).
        cute.arch.fence_acq_rel_sys()

    @cute.jit
    def _gin_return_and_wait(self, token_comm_args, lane_idx):
        """C5 + C6 on CTA0 warp 0: return node partials, wait for ours."""
        from nccl.core.device import cute as nccl_cute

        args = token_comm_args
        num_slots: cutlass.Constexpr[int] = self.num_remote_slots
        t_max: cutlass.Constexpr[int] = self.max_tokens_per_rank
        row_bytes: cutlass.Constexpr[int] = self.hidden * 2
        if lane_idx < Int32(num_slots):
            slot = lane_idx
            n_s = Int32(
                cute.arch.load(
                    args.gin_token_count.iterator + (Int32(1) + slot), Int32,
                    sem="relaxed", scope="gpu",
                )
            )
            gin = self._gin()
            win = args.gin_window
            team = self._team_rail_value
            coop = nccl_cute.thread()
            dst_node = self._remote_slot_node(slot)
            my_slot = self._slot_at_node(dst_node)
            ret_signal = Int32(self.gin_ret_signal_base) + my_slot
            if n_s > Int32(0):
                nbytes = n_s * Int32(row_bytes)
                src = self._gin_bytes(
                    args, args.gin_reduce_buf.iterator.toint()
                    + Int64(slot) * Int64(t_max * row_bytes),
                    nbytes,
                )
                dst = self._gin_bytes(
                    args, args.gin_remote_partial.iterator.toint()
                    + Int64(my_slot) * Int64(t_max * row_bytes),
                    nbytes,
                )
                gin.put(
                    team, dst_node, win, dst, win, src, coop,
                    is_signal=True, signal_id=ret_signal,
                    signal_op=_GIN_SIGNAL_OP_ADD, signal_op_arg=Int64(1),
                )
            else:
                gin.signal(
                    team, dst_node, True, ret_signal, _GIN_SIGNAL_OP_ADD,
                    Int64(1), coop,
                )
        cute.arch.sync_warp()
        if lane_idx == Int32(0):
            for s in cutlass.range_constexpr(num_slots):
                self._gin_wait_signal(self.gin_ret_signal_base + s, 1)
        cute.arch.sync_warp()

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
            cta_linear_id = self._cta_linear_id()
            local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)
            self._dbg(cta_linear_id, 20, local_warp_idx, lane_idx)
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
            if cutlass.const_expr(self.enable_nccl_gin):
                self._dbg(cta_linear_id, 21, local_warp_idx, lane_idx)
                # B1 above made every node-local FC2 write visible, including
                # the landing rows of the remote origins this rank proxies.
                self._gin_node_reduce(
                    token_comm_args, cta_linear_id, local_warp_idx, lane_idx,
                )
                self._dbg(cta_linear_id, 22, local_warp_idx, lane_idx)
                software_grid_sync(
                    token_comm_args.grid_sync_counter,
                    cta_linear_id,
                    token_comm_args.sm_count,
                    local_warp_idx * Int32(self.warp_threads) + lane_idx,
                    num_threads=self.num_dispatch_threads,
                )
                self._dbg(cta_linear_id, 23, local_warp_idx, lane_idx)
                if cta_linear_id == Int32(0) and local_warp_idx == Int32(0):
                    self._gin_return_and_wait(token_comm_args, lane_idx)
                    self._dbg(cta_linear_id, 25, local_warp_idx, lane_idx)
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
            self._dbg(cta_linear_id, 27, local_warp_idx, lane_idx)
            if cutlass.const_expr(self.enable_nccl_gin):
                # Every CTA read the launch epoch long before the final grid
                # sync above; the next launch waits for payload flags epoch+2.
                if cta_linear_id == Int32(0) and local_warp_idx == Int32(0):
                    if lane_idx == Int32(0):
                        cute.arch.atomic_add(
                            token_comm_args.gin_epoch.iterator, Int32(1),
                            sem="relaxed", scope="gpu",
                        )
