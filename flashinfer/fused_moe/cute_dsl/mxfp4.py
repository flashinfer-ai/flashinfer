# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Planned MXFP4 W4A8 routed MoE; minimum architecture SM100 (B300: SM103).

Planning binds caller-owned buffers and compiles the selected offline tactic.
Execution reads their current contents, including runtime SiTU parameters.
Use a separate plan/output/workspace for concurrently executing calls.

Rank-local layouts come from explicit metadata (``Mxfp4MoEParallelLayout`` or
the explicit local expert interval), never from tensor shapes. Expert
parallelism owns a contiguous global expert interval; MoE tensor parallelism
owns an intermediate-dimension shard of every expert. Hybrid layouts are
rejected. Every rank output is a partial sum whose reduction is external.
"""

from dataclasses import dataclass
from math import prod
from typing import Optional, Union

import cuda.bindings.driver as cuda
import os
import torch

from ...tllm_enums import ActivationType
from .fused_moe import _moe_core_impl, validate_w4a8_inputs
from .moe_utils import get_max_num_tiles, moe_sort
from .mxfp4_finalize import plan_finalize_rows
from .mxfp4_routing import FUSED_ROUTE_MAX_ROUTES, _plan_route_preprocess
from .blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    blockscaled_contiguous_gather_grouped_gemm_act_fusion,
)
from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    blockscaled_contiguous_grouped_gemm_finalize_fusion,
)
from .swapab_moe import (
    SWAP_ROW_TILE,
    fill_permuted_token_index,
    swap_row_tma,
    swapab_dispatch,
    swapab_gemm1_situ,
    swapab_gemm2,
)
from .tuner import DEFAULT_BLACKWELL_MOE_TACTIC, canonicalize_w4a8_tactic

# Swap-AB finalize form by token count. T <= this bound reduce-adds
# ``alpha * route_weight * acc`` straight from the GEMM2 epilogue
# (``red.global.add.bf16x2`` into the zero-filled output). Above it GEMM2
# writes ``alpha * acc`` rows in permuted order into workspace and a separate
# gather kernel applies the route weights: the per-element reductions cost
# ~80-160 G adds/s, which at TP8 T=1024 (8192 local rows x 7168) added
# ~380 us to a 244 us GEMM2, while the two-stage form streams the rows once.
SWAP_ATOMIC_FINALIZE_MAX_TOKENS = int(
    os.environ.get("SWAPAB_ATOMIC_FINALIZE_MAX_TOKENS", "16")
)
# Two-stage finalize only pays off with a narrow GEMM2 K (short per-tile
# mainloops cannot hide the epilogue's reductions): TP8 (K=384) T=128 saved
# 38 us, EP8 (K=3072) T=128..1024 lost 6-28 us against the fused epilogue.
SWAP_TWO_STAGE_MAX_SHARD = int(os.environ.get("SWAPAB_TWO_STAGE_MAX_SHARD", "512"))
# Largest token count whose swap-AB GEMM2 uses 4-K-block stages (deeper
# pipeline); above it the intermediate shard streams in one 12-block stage.
SWAP_GEMM2_SHORT_STAGE_MAX_TOKENS = int(
    os.environ.get("SWAPAB_GEMM2_SHORT_STAGE_MAX_TOKENS", "256")
)
# L2 eviction policy for the dense grouped GEMMs' weight TMA loads (CUTLASS
# SM90 TMA cache-hint encodings). Weights stream once per token tile while the
# gathered activations are re-read by every N tile of an expert: EVICT_FIRST
# keeps the activations resident (B300, TP8 T=2048 balanced: GEMM1 530 ->
# 511 us, GEMM2 345 -> 316 us; T=4096: 1049 -> 930 us total).
_DENSE_L2_HINTS = {
    "none": None,
    "first": 0x12F0000000000000,
    "last": 0x14F0000000000000,
}
DENSE_WEIGHT_L2_HINT = _DENSE_L2_HINTS[os.environ.get("MXFP4_DENSE_L2HINT", "first")]
# Hybrid prefill form (wide swap tiles, finalize=True): the swap GEMM1 runs
# ``n_tile``-row sub-tiles of 128-row sort groups (only the occupied ones, via
# a device-built work list) and writes its MXFP8 rows with the block-scaled
# SFA layout, so the dense contiguous grouped GEMM2 with the bulk-reduce
# finalize consumes them directly: no permuted partial rows, no finalize
# kernel. B300 TP8 T=2048: 434 + 314 us against 434 + 295 + 94 (two-stage).
SWAP_HYBRID = os.environ.get("SWAPAB_HYBRID", "1") != "0"
SWAP_HYBRID_GROUP_ROWS = 128
SWAP_HYBRID_MIN_TILE = 64
# Hybrid form, mixed GEMM1 tiles: a 128-row sort group with more valid rows
# than this runs as one dense gather-GEMM1 tile (its expert's weights stream
# once for the group); the other groups run as swap sub-tiles, which stream
# the weights once per sub-tile. Measured on B300 for the MoE-TP shard: 16
# hot experts holding every route at T=128..2048 re-stream each expert 16x
# in the swap form (148 us at T=128 against 53 us for the dense form). 128
# disables the dense tiles.
SWAP_HYBRID_DENSE_MIN_ROWS = int(os.environ.get("SWAPAB_HYBRID_DENSE_MIN_ROWS", "64"))
# Mixed form below the hybrid cap (finalize only, T >= SWAP_MIXED_MIN_TOKENS):
# 128-row sort groups as in the hybrid form, dense gather-GEMM1 tiles for the
# groups above SWAP_HYBRID_DENSE_MIN_ROWS and swap GEMM1 sub-tiles for the
# rest, but the swap GEMM2 of the policy tile over every occupied sub-tile
# (reading the blocked scales) instead of the dense finalize GEMM2. Targets
# the concentrated routings (few experts holding every route) of the MoE-TP
# shard at T=128..1024; SWAPAB_MIXED=1 enables it, SWAPAB_MIXED_EP=1 also on
# expert-parallel ranks.
SWAP_MIXED = os.environ.get("SWAPAB_MIXED", "0") == "1"
SWAP_MIXED_MIN_TOKENS = int(os.environ.get("SWAPAB_MIXED_MIN_TOKENS", "128"))
SWAP_MIXED_EP = os.environ.get("SWAPAB_MIXED_EP", "0") == "1"
# Mixed form: dense tiles only when the full groups hold at least this share
# (per mille) of the valid rows (0 = always); timing-only switch to keep the
# swap GEMM1/GEMM2 scales plain (valid with the dense tiles disabled).
SWAP_MIXED_WIDE_PERMILLE = int(os.environ.get("SWAPAB_MIXED_WIDE_PERMILLE", "0"))
SWAP_MIXED_SF_PLAIN = os.environ.get("SWAPAB_MIXED_SF_PLAIN", "0") == "1"
# Mixed form: weight M-tiles per swap-GEMM2 work item. Measured on B300 (TP8
# T=256/1024 balanced): with the 128-row groups the GEMM2 of the policy tile
# loses 6-9 % at m_group 1 and is back at the 32-row-group time with 2.
SWAP_MIXED_GEMM2_MGROUP = int(os.environ.get("SWAPAB_MIXED_MGROUP2", "2"))
# Swap GEMM2 of the narrow MoE-TP shard (K = 384): two weight M-tiles per
# work item with 128-wide stages. Measured on B300 (TP8 rank 0, same GPU):
# every row from T=4 up is faster (balanced/hot -0.5..-4.6 %, empty
# -2..-7.5 %: T=1024 empty 388 -> 359 us, T=16 hot 182 -> 173 us), T=1 is
# unchanged; the wide expert-parallel shard (K = 3072) loses 7-11 % on its
# 24-40 us decode/empty rows with the same grouping and keeps one tile.
# Below SWAP_TP_GEMM2_MGROUP_MIN_TOKENS the shard keeps one tile too: with a
# single active expert (``empty`` routing) GEMM2 has 56 weight tiles and the
# grouping halves the CTAs that stream them (+0.3..0.8 us on 26-37 us rows)
# while the balanced/hot rows gain 1-5 us; the acceptance rule of this work
# forbids a slower row, so the grouping starts where every row gains.
# SWAPAB_MGROUP2 (swapab_moe.swap_m_group) still overrides both.
SWAP_TP_GEMM2_MGROUP = int(os.environ.get("SWAPAB_TP_MGROUP2", "2"))
SWAP_TP_GEMM2_MGROUP_MIN_TOKENS = int(
    os.environ.get("SWAPAB_TP_MGROUP2_MIN_TOKENS", "16")
)
# Swap-AB path: EVICT_FIRST on the weight loads helps every shape whose
# experts stream their weights once (decode -3..-15 us, single-group prefill
# -5..-14 us on B300, same-GPU pairs) and hurts a hot expert whose groups
# re-read them (EP8 T=128 hot +10 us, balanced +4 us).  The policy below is
# per layout and token count (see ``_swap_weight_l2_hint``): every layout
# uses it for decode; an expert-parallel rank skips it inside
# ``SWAP_EP_L2HINT_SKIP`` where the hot rows are thin.
SWAP_WEIGHT_L2_HINT_MAX_TOKENS = int(os.environ.get("SWAPAB_L2HINT_MAX_TOKENS", "16"))
SWAP_EP_L2HINT_SKIP = (
    SWAP_WEIGHT_L2_HINT_MAX_TOKENS + 1,
    int(os.environ.get("SWAPAB_EP_L2HINT_SKIP_MAX_TOKENS", "255")),
)
# Dense-path (T > swapab_max_tokens) W4A8 tactic measured on B300 (Kimi K3):
# 256-wide MMA N halves the GEMM2 tile count (TP8 T=2048 GEMM2 444 -> 344 us,
# T=4096 564 -> 459 us) and matches GEMM1; used when no offline table covers T.
B300_SITU_DENSE_TACTIC = (128, ((128, 256), (1, 1), False), ((128, 256), (1, 1), False))
# Dense-path W4A8 tactics by token-count bucket, measured on B300 (Kimi K3,
# candidate-only CUPTI graph medians, every tactic of a bucket timed in the
# same process on the same GPU). The rows per local expert set the M-tile
# padding. Every bucket keeps the M128 GEMM1 tile: on the narrow MoE-TP
# shard the 2-CTA M256 tactic measured best at T=8192 (every routing keeps
# >= 146 rows per expert; GEMM1 905 -> 774 us), but compute-sanitizer
# synccheck reports "Missing wait" records from that kernel's MMA warp, so
# the shard runs the cluster-2 N256 GEMM2 with the M128 GEMM1 up to T=8192
# instead (T=8192 balanced/hot/empty -0.4/-3.7/-0.8% against the base
# tactic). The expert-parallel rank would gain 17% from M256 on the balanced
# routing at T=8192 but lose 31% on the remote-dominated one (73 rows per
# expert, padded 3.5x); a routing-aware tile choice needs both tile lists at
# run time and is left open. The 192-wide GEMM2 helps where GEMM2 is
# finalize-bound: cluster 1 wins 1-2% on the wide shard up to T=4096 and
# cluster 2 wins 3-5% there at T=16384 (not at 32768, where the default is
# best); on the narrow shard the cluster-2 N192 GEMM2 wins 1-7% on every
# routing from T=16384 (16384: -3.4..-4.6%, 32768: -1.2..-7.2%).
_T128_N256_C2 = (128, ((128, 256), (1, 1), False), ((128, 256), (1, 2), False))
_T128_N192 = (128, ((128, 256), (1, 1), False), ((128, 192), (1, 1), False))
_T128_N192_C2 = (128, ((128, 256), (1, 1), False), ((128, 192), (1, 2), False))
B300_SITU_DENSE_TACTIC_TABLE_WIDE = (
    (8192, _T128_N192),
    (16384, _T128_N192_C2),
    (1 << 62, B300_SITU_DENSE_TACTIC),
)
B300_SITU_DENSE_TACTIC_TABLE_NARROW = (
    (2048, B300_SITU_DENSE_TACTIC),
    (8192, _T128_N256_C2),
    (1 << 62, _T128_N192_C2),
)
# Experimental: dense-path GEMM2 writes expanded rows and ``moe_unpermute``
# applies the route weights (no bulk reduce-add into the output).
DENSE_TWO_STAGE_FINALIZE = os.environ.get("MXFP4_DENSE_TWO_STAGE", "0") == "1"


_PARALLEL_MODES = ("single", "expert_parallel", "moe_tensor_parallel")


@dataclass(frozen=True)
class Mxfp4MoEParallelLayout:
    """Explicit rank placement; ``size`` ranks, this process is ``rank``.

    ``mode`` is ``"single"`` (one rank owns everything), ``"expert_parallel"``
    (rank ``r`` owns global experts ``[r*E/size, (r+1)*E/size)`` with the
    full intermediate dimension) or ``"moe_tensor_parallel"`` (every rank owns
    all experts and intermediate columns ``[r*I/size, (r+1)*I/size)`` of each).
    Hybrid expert/tensor parallelism is not representable; ``from_sizes``
    rejects it. Global ``num_experts`` and ``intermediate_size`` are resolved
    into rank-local values by ``resolve_mxfp4_moe_layout``.
    """

    mode: str = "single"
    size: int = 1
    rank: int = 0

    def __post_init__(self):
        if self.mode not in _PARALLEL_MODES:
            raise ValueError(f"parallel mode must be one of {_PARALLEL_MODES}")
        if not isinstance(self.size, int) or isinstance(self.size, bool):
            raise ValueError("parallel size must be an int")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise ValueError("parallel rank must be an int")
        if self.size < 1:
            raise ValueError("parallel size must be positive")
        if not 0 <= self.rank < self.size:
            raise ValueError(f"parallel rank must be in [0, {self.size})")
        if self.mode == "single" and (self.size, self.rank) != (1, 0):
            raise ValueError("single layout requires size 1 and rank 0")

    @classmethod
    def from_sizes(
        cls,
        *,
        ep_size: int = 1,
        ep_rank: int = 0,
        moe_tp_size: int = 1,
        moe_tp_rank: int = 0,
    ) -> "Mxfp4MoEParallelLayout":
        """Build a layout from EP/TP sizes; both above one is a hybrid error."""
        for name, size, rank in (
            ("ep", ep_size, ep_rank),
            ("moe_tp", moe_tp_size, moe_tp_rank),
        ):
            if size < 1 or not 0 <= rank < size:
                raise ValueError(
                    f"require {name}_size >= 1 and 0 <= {name}_rank < size"
                )
        if ep_size > 1 and moe_tp_size > 1:
            raise ValueError(
                "hybrid expert/tensor parallelism is unsupported: "
                f"ep_size={ep_size} and moe_tp_size={moe_tp_size} both exceed 1"
            )
        if ep_size > 1:
            return cls("expert_parallel", ep_size, ep_rank)
        if moe_tp_size > 1:
            return cls("moe_tensor_parallel", moe_tp_size, moe_tp_rank)
        return cls()


@dataclass(frozen=True)
class Mxfp4MoERankLayout:
    """Resolved rank-local geometry; ``num_experts``/``intermediate_size`` are global.

    ``parallel_size``/``parallel_rank`` are ``None`` when an explicit expert
    interval does not coincide with a uniform expert-parallel rank slice.
    """

    mode: str
    num_experts: int
    intermediate_size: int
    num_local_experts: int
    local_expert_offset: int
    intermediate_shard: int
    parallel_size: Optional[int] = None
    parallel_rank: Optional[int] = None

    @property
    def gemm1_n(self) -> int:
        """Rank-local GEMM1 output width: interleaved up and gate rows."""
        return 2 * self.intermediate_shard

    @property
    def gemm2_k(self) -> int:
        """Rank-local GEMM2 contraction length."""
        return self.intermediate_shard


def resolve_mxfp4_moe_layout(
    num_experts: int,
    intermediate_size: int,
    *,
    parallel_layout: Optional[Mxfp4MoEParallelLayout] = None,
    num_local_experts: Optional[int] = None,
    local_expert_offset: Optional[int] = None,
) -> Mxfp4MoERankLayout:
    """Derive and validate rank-local geometry from explicit metadata only.

    ``parallel_layout`` is the uniform EP/TP form. ``num_local_experts`` and
    ``local_expert_offset`` are the explicit expert-interval form, which is
    expert parallelism (or single when the interval is every expert). When
    both forms are given they must agree. Nothing is inferred from tensors.
    """
    if num_experts <= 0 or intermediate_size <= 0:
        raise ValueError("num_experts and intermediate_size must be positive")
    if parallel_layout is None:
        local = num_experts if num_local_experts is None else num_local_experts
        offset = 0 if local_expert_offset is None else local_expert_offset
        if local <= 0 or offset < 0 or offset + local > num_experts:
            raise ValueError(
                "local experts must form a nonempty contiguous global expert interval"
            )
        if local == num_experts and offset == 0:
            return Mxfp4MoERankLayout(
                "single",
                num_experts,
                intermediate_size,
                local,
                0,
                intermediate_size,
                1,
                0,
            )
        uniform = num_experts % local == 0 and offset % local == 0
        return Mxfp4MoERankLayout(
            "expert_parallel",
            num_experts,
            intermediate_size,
            local,
            offset,
            intermediate_size,
            num_experts // local if uniform else None,
            offset // local if uniform else None,
        )
    if not isinstance(parallel_layout, Mxfp4MoEParallelLayout):
        raise TypeError("parallel_layout must be an Mxfp4MoEParallelLayout")
    mode, size, rank = parallel_layout.mode, parallel_layout.size, parallel_layout.rank
    if mode == "expert_parallel":
        if num_experts % size:
            raise ValueError(
                f"expert parallelism requires num_experts ({num_experts}) divisible "
                f"by ep size ({size})"
            )
        local, offset, shard = (
            num_experts // size,
            rank * (num_experts // size),
            (intermediate_size),
        )
    elif mode == "moe_tensor_parallel":
        if intermediate_size % size or (intermediate_size // size) % 128:
            raise ValueError(
                f"MoE tensor parallelism requires intermediate_size ({intermediate_size}) "
                f"divisible by moe_tp size ({size}) into a multiple of 128"
            )
        local, offset, shard = num_experts, 0, intermediate_size // size
    else:
        local, offset, shard = num_experts, 0, intermediate_size
    if num_local_experts is not None and num_local_experts != local:
        raise ValueError(
            f"num_local_experts={num_local_experts} is inconsistent with "
            f"{mode} size {size} rank {rank}, which owns {local} local experts"
        )
    if local_expert_offset is not None and local_expert_offset != offset:
        raise ValueError(
            f"local_expert_offset={local_expert_offset} is inconsistent with "
            f"{mode} size {size} rank {rank}, whose offset is {offset}"
        )
    return Mxfp4MoERankLayout(
        mode, num_experts, intermediate_size, local, offset, shard, size, rank
    )


@dataclass(frozen=True)
class Mxfp4MoECapability:
    """Support verdict plus the resolved rank-local layout when it resolves."""

    supported: bool
    reason: str
    cuda_graph: bool
    layout: Optional[Mxfp4MoERankLayout] = None
    # Deferred finalize (``plan(..., do_finalize=False)``): rank-local GEMM2
    # rows plus the route weights and the assignment-to-row map, for callers
    # fusing the route-weight reduction with their own collectives.
    deferred_output: bool = False


def mxfp4_moe_capability(
    *,
    gpu_arch: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    local_expert_offset: Optional[int] = None,
    parallel_layout: Optional[Mxfp4MoEParallelLayout] = None,
    quantization: str = "mxfp4_w4a8",
    activation_type: ActivationType = ActivationType.Situ,
    cuda_graph: bool = True,
    do_finalize: bool = True,
) -> Mxfp4MoECapability:
    """Query support from explicit metadata without allocating or using CUDA.

    ``gpu_arch`` is 100 for SM100 or 103 for SM103. Weight scales are UE8M0
    with group size 32; activation storage is E4M3 and output is BF16.
    ``num_experts`` and ``intermediate_size`` are global model values; the
    rank-local expert interval and intermediate shard are derived from
    ``parallel_layout`` and/or the explicit ``num_local_experts`` and
    ``local_expert_offset`` (see ``resolve_mxfp4_moe_layout``). The result
    carries that resolved layout; CUDA Graph capture is supported for every
    supported configuration in both parallel modes. ``do_finalize=False``
    asks for the deferred output form (GEMM2 rows in permuted order, route
    weights and the expanded->permuted row map), which the SiTU swap-AB
    path provides at every token count; ``deferred_output`` reports it.
    """
    reason = ""
    layout = None
    deferred = activation_type == ActivationType.Situ
    if gpu_arch not in (100, 103):
        reason = "MXFP4 W4A8 requires SM100 or SM103"
    elif quantization != "mxfp4_w4a8":
        reason = "quantization must be explicitly mxfp4_w4a8"
    elif activation_type not in (ActivationType.Situ, ActivationType.Swiglu):
        reason = "planned MXFP4 supports SiTU and SwiGLU"
    elif min(hidden_size, intermediate_size) <= 0 or (
        hidden_size % 128 or intermediate_size % 128
    ):
        reason = "hidden and intermediate dimensions must be positive multiples of 128"
    elif not (1 <= top_k <= num_experts <= 1024):
        reason = "require 1 <= top_k <= num_experts <= 1024"
    elif top_k > 32:
        reason = "top_k must not exceed 32"
    else:
        try:
            layout = resolve_mxfp4_moe_layout(
                num_experts,
                intermediate_size,
                parallel_layout=parallel_layout,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
            )
        except (TypeError, ValueError) as error:
            reason = str(error)
    if not reason and not do_finalize and not deferred:
        reason = "deferred (do_finalize=False) output requires SiTU activation"
    return Mxfp4MoECapability(
        not reason, reason, not reason, layout, not reason and deferred
    )


@dataclass(frozen=True)
class _WorkspaceField:
    name: str
    shape: tuple
    dtype: torch.dtype
    offset: int
    nbytes: int


def _align(size: int) -> int:
    return (size + 255) // 256 * 256


def _byte_interval(tensor):
    first = tensor.data_ptr()
    span = 1 + sum(
        (dim - 1) * stride
        for dim, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return first, first + span * tensor.element_size()


def _overlap(left, right):
    return max(left[0], right[0]) < min(left[1], right[1])


class Mxfp4MoEPlan:
    """Executable with fixed buffer addresses, prepared outside graph capture.

    Update bound activation, routing, beta and scale tensors in-place before
    calling ``run`` or replaying a captured graph. No weights are copied.
    The output is this rank's partial sum: its local experts under expert
    parallelism, or its intermediate shard under MoE tensor parallelism.
    """

    def __init__(
        self, *, kwargs, workspace, topk_ids, topk_weights, route_ids, route_weights
    ):
        # Keep every bound tensor alive alongside the raw launch pointers.
        self._kwargs = kwargs
        self.workspace = workspace
        self.output = kwargs["moe_output"]
        self._topk_ids = topk_ids
        self._topk_weights = topk_weights
        self._route_ids = route_ids
        self._route_weights = route_weights
        self._route_preprocess = None
        self.device = self.output.device
        self._packed_weight_view = (
            topk_ids.view(torch.bfloat16)[:, ::2] if topk_weights is None else None
        )

    def _prepare_routing(self):
        if self._topk_weights is None:
            torch.bitwise_right_shift(self._topk_ids, 16, out=self._route_ids)
            self._route_weights.copy_(self._packed_weight_view)
        elif self._route_weights is not self._topk_weights:
            self._route_weights.copy_(self._topk_weights)

    def _prepare(self):
        # The existing path validates and warms the exact pointers/callables
        # retained below. No stream is retained: run resolves the caller's.
        launches = {}
        with torch.cuda.device(self.device):
            self._prepare_routing()
            _moe_core_impl(**self._kwargs, _prepared_launches=launches)
        self._sort, self._sort_args = launches["sort"]
        self._gather, self._gather_args, self._gather_kwargs = launches["gather"]
        # No memset in the expanded-row (non-fused) finalize form.
        self._memset, self._memset_args = launches.get("memset", (None, None))
        self._finalize, self._finalize_args = launches["finalize"]
        self._unpermute = launches.get("unpermute")
        self._finalize_rows = None
        if "gemm2_partial" in launches:
            self._finalize_rows = plan_finalize_rows(
                launches["gemm2_partial"],
                self._kwargs["moe_sort_buffers"]["out_expanded_idx_to_permuted_idx"],
                self._route_weights,
                self.output,
                expanded_rows=True,
            )
        if self.output.shape[0] <= 16:
            self._route_preprocess = _plan_route_preprocess(
                self._topk_ids,
                self._topk_weights,
                route_ids=self._route_ids,
                route_weights=self._route_weights,
                output=self.output,
                moe_sort_buffers=(
                    None
                    if self._kwargs["enable_pdl"]
                    else self._kwargs["moe_sort_buffers"]
                ),
                num_experts=self._kwargs["num_experts"],
                num_local_experts=self._kwargs["num_local_experts"],
                local_expert_offset=self._kwargs["local_expert_offset"],
                tile_size=self._kwargs["tile_size"],
                _single_tile_per_expert=self._kwargs.get(
                    "_enable_decode_specialization", False
                ),
            )
            # Preprocessing warmup clears output. Finish the complete MoE so
            # plan retains its existing valid-output postcondition.
            self.run()
        elif not self._kwargs["enable_pdl"]:
            # T > 16: one conversion + output-clear launch replaces the torch
            # unpack kernels and the separate memset (run() skips the memset
            # whenever a route preprocess is bound).
            self._route_preprocess = _plan_route_preprocess(
                self._topk_ids,
                self._topk_weights,
                route_ids=self._route_ids,
                route_weights=self._route_weights,
                output=self.output,
            )
            self.run()

    def run(self) -> torch.Tensor:
        """Enqueue on the caller's current stream and return the bound output.

        All GPU buffers and compiled kernels were prepared by ``plan``.
        This method performs no tuning, allocation, or host synchronization.
        """
        with torch.cuda.device(self.device):
            stream_ptr = torch.cuda.current_stream().cuda_stream
            stream = cuda.CUstream(stream_ptr)
            if self._route_preprocess is None:
                self._prepare_routing()
            else:
                self._route_preprocess.run(stream)
            if (
                self._route_preprocess is None
                or not self._route_preprocess.sorts_tokens
            ):
                self._sort(*self._sort_args, stream_ptr)
            self._gather(*self._gather_args, stream=stream, **self._gather_kwargs)
            if self._route_preprocess is None and self._memset is not None:
                self._memset(*self._memset_args, stream_ptr)
            self._finalize(*self._finalize_args, stream=stream)
            if self._unpermute is not None:
                unpermute, unpermute_kwargs = self._unpermute
                unpermute(**unpermute_kwargs)
            if self._finalize_rows is not None:
                self._finalize_rows.run(stream)
        return self.output


class Mxfp4MoESwapAbPlan:
    """Plan on the swap-AB path: route preprocessing (ID unpack, FP32 weights,
    output zero-fill; fused with ``n_tile``-row expert grouping for T <= 16,
    followed by ``moe_sort`` above) and the two swap-AB grouped GEMMs. Same
    contract as :class:`Mxfp4MoEPlan`: fixed buffer addresses,
    graph-capturable ``run``. ``finalize=False`` is the deferred form: GEMM2
    writes ``alpha * acc`` rows in permuted order into ``output`` and the plan
    exposes ``expanded_idx_to_permuted_idx`` / ``route_weights``. With
    ``finalize=True`` and T > ``SWAP_ATOMIC_FINALIZE_MAX_TOKENS`` the same
    permuted rows go to the ``partial_rows`` workspace region and a fifth
    launch (:func:`plan_finalize_rows`) applies the route weights
    (``two_stage``); at or below the bound GEMM2 reduce-adds into ``output``.
    """

    def __init__(
        self,
        *,
        wrapper,
        buffers,
        workspace,
        x,
        x_sf,
        topk_ids,
        topk_weights,
        w1,
        w1_sf,
        w2,
        w2_sf,
        beta,
        linear_beta,
        output,
        n_tile,
        finalize=True,
    ):
        self._wrapper = wrapper
        self._buffers = buffers
        self.workspace = workspace
        self.output = output
        self.device = output.device
        self.n_tile = n_tile
        self.finalize = bool(finalize)
        self.deferred = not self.finalize
        self.hybrid = wrapper._swap_hybrid(x.shape[0], self.finalize)
        self.mixed = wrapper._swap_mixed(x.shape[0], self.finalize)
        self.group_rows = (
            SWAP_HYBRID_GROUP_ROWS if (self.hybrid or self.mixed) else n_tile
        )
        self.two_stage = (
            self.finalize
            and not self.hybrid
            and x.shape[0] > SWAP_ATOMIC_FINALIZE_MAX_TOKENS
            and wrapper.intermediate_shard <= SWAP_TWO_STAGE_MAX_SHARD
        )
        self._partial_rows = buffers["partial_rows"] if self.two_stage else None
        self._finalize_rows = None
        # Deferred-finalize outputs (valid after ``run``): row of each
        # (token, slot) assignment (-1 when not local) and FP32 route weights.
        self.expanded_idx_to_permuted_idx = buffers["out_expanded_idx_to_permuted_idx"]
        self._inputs = (x, x_sf, topk_ids, topk_weights, w1, w1_sf, w2, w2_sf)
        self._beta = beta
        self._linear_beta = linear_beta
        # Same private surface as Mxfp4MoEPlan (tests poke these buffers).
        self._topk_ids = topk_ids
        self._topk_weights = topk_weights
        self._kwargs = {
            "gemm1_out": buffers["gemm1_out"],
            "gemm1_out_scale": buffers["gemm1_out_scale"],
            "moe_output": output,
            "moe_sort_buffers": {
                name: value
                for name, value in buffers.items()
                if name.startswith("out_")
            },
        }
        self._route_ids = buffers["route_ids"] if topk_weights is None else topk_ids
        self._route_weights = (
            topk_weights
            if topk_weights is not None and topk_weights.dtype == torch.float32
            else buffers["route_weights"]
        )
        self.route_weights = self._route_weights

    def _prepare_routing(self):
        if self._topk_weights is None:
            torch.bitwise_right_shift(self._topk_ids, 16, out=self._route_ids)
            self._route_weights.copy_(self._packed_weight_view)
        elif self._route_weights is not self._topk_weights:
            self._route_weights.copy_(self._topk_weights)

    def _prepare(self):
        w = self._wrapper
        x, x_sf, topk_ids, topk_weights, w1, w1_sf, w2, w2_sf = self._inputs
        b = self._buffers
        num_tokens = x.shape[0]
        self._route_preprocess = None
        self._sort = None
        self._dispatch = None
        self._dispatch_args = None
        self._gemm1_dense = None
        self._token_index = None
        self._token_index_args = None
        self._packed_weight_view = (
            topk_ids.view(torch.bfloat16)[:, ::2] if topk_weights is None else None
        )
        sort_buffers = {
            name: value for name, value in b.items() if name.startswith("out_")
        }
        launches = {}
        # The route preprocess kernel clears the finalize output; in the
        # deferred form the first T rows of the row buffer stand in (they are
        # overwritten or padding, so the clear is harmless).
        # In the deferred and two-stage forms the first T permuted rows stand
        # in (overwritten or padding, so the clear is harmless): the finalize
        # kernel writes every output row itself.
        if self.finalize and not self.two_stage:
            clear_target = self.output
        elif self.two_stage:
            clear_target = self._partial_rows[:num_tokens]
        else:
            clear_target = self.output[:num_tokens]
        with torch.cuda.device(self.device):
            if num_tokens * w.top_k <= FUSED_ROUTE_MAX_ROUTES:
                # Fused routing (T <= 256 for top-16): ID unpack, FP32
                # weights, n_tile-row groups and the output zero-fill in one
                # single-CTA launch instead of the conversion kernel plus
                # ``moe_sort`` (about 3 us against 11 us at T=128).
                self._route_preprocess = _plan_route_preprocess(
                    topk_ids,
                    topk_weights,
                    route_ids=self._route_ids,
                    route_weights=self._route_weights,
                    output=clear_target,
                    moe_sort_buffers=sort_buffers,
                    num_experts=w.num_experts,
                    num_local_experts=w.num_local_experts,
                    local_expert_offset=w.local_expert_offset,
                    tile_size=self.group_rows,
                    _single_tile_per_expert=self.group_rows >= num_tokens,
                    clear_output=not self.two_stage,
                )
            else:
                # Generic routing: one conversion + output-clear launch, then
                # moe_sort with n_tile-row groups.
                self._route_preprocess = _plan_route_preprocess(
                    topk_ids,
                    topk_weights,
                    route_ids=self._route_ids,
                    route_weights=self._route_weights,
                    output=clear_target,
                    clear_output=not self.two_stage,
                )
                moe_sort(
                    token_selected_experts=self._route_ids,
                    token_final_scales=self._route_weights,
                    num_experts=w.num_experts,
                    top_k=w.top_k,
                    local_expert_offset=w.local_expert_offset,
                    num_local_experts=w.num_local_experts,
                    tile_tokens_dim=self.group_rows,
                    enable_pdl=w.enable_pdl,
                    _prepared_launches=launches,
                    **sort_buffers,
                )
                self._sort, self._sort_args = launches["sort"]
            gemm1_lists = {}
            if self.hybrid or self.mixed:
                # Work lists over the 128-row sort groups: wide (dense GEMM1
                # tiles), narrow (swap GEMM1 sub-tiles) and, in the mixed
                # form, every occupied sub-tile for the swap GEMM2.
                swapab_dispatch(
                    tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
                    num_non_exiting_tiles=b["out_num_non_exiting_tiles"],
                    group_rows=self.group_rows,
                    narrow_tile=self.n_tile,
                    wide_list=b["swap_wide_list"],
                    wide_count=b["swap_wide_count"],
                    narrow_list=b["swap_row_groups"],
                    narrow_count=b["swap_row_group_count"],
                    wide_min_rows=min(SWAP_HYBRID_DENSE_MIN_ROWS, self.group_rows),
                    wide_min_permille=SWAP_MIXED_WIDE_PERMILLE if self.mixed else 0,
                    all_list=b["swap_all_groups"] if self.mixed else None,
                    all_count=b["swap_all_count"] if self.mixed else None,
                    enable_pdl=w.enable_pdl,
                    _prepared_launches=launches,
                )
                self._dispatch, self._dispatch_args = launches["swap_dispatch"]
                gemm1_lists = dict(
                    tile_idx_to_row_group=b["swap_row_groups"],
                    num_non_exiting_tiles=b["swap_row_group_count"],
                    group_rows=self.group_rows,
                    sf_blocked=not (self.mixed and SWAP_MIXED_SF_PLAIN),
                )
            token_idx = None
            if swap_row_tma(self.n_tile, True):
                fill_permuted_token_index(
                    b["out_permuted_idx_to_expanded_idx"],
                    b["permuted_idx_to_token_idx"],
                    num_tokens,
                    w.top_k,
                    _prepared_launches=launches,
                )
                self._token_index, self._token_index_args = launches["swap_token_index"]
                token_idx = b["permuted_idx_to_token_idx"]
            swapab_gemm1_situ(
                w1=w1,
                w1_sf=w1_sf,
                x=x,
                x_sf=x_sf,
                permuted_idx_to_token_idx=token_idx,
                permuted_idx_to_expanded_idx=b["out_permuted_idx_to_expanded_idx"],
                act=b["gemm1_out"],
                act_sf=b["gemm1_out_scale"],
                tile_idx_to_expert_idx=b["out_tile_idx_to_expert_idx"],
                tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
                alpha=b["w1_alpha"],
                beta=self._beta,
                linear_beta=self._linear_beta,
                top_k=w.top_k,
                zero_output=None,
                n_tile=self.n_tile,
                enable_pdl=w.enable_pdl,
                weight_l2_hint=w._swap_weight_l2_hint(num_tokens),
                _prepared_launches=launches,
                **{
                    "num_non_exiting_tiles": b["out_num_non_exiting_tiles"],
                    **gemm1_lists,
                },
            )
            if (
                self.hybrid or self.mixed
            ) and self.group_rows > SWAP_HYBRID_DENSE_MIN_ROWS:
                # Dense gather GEMM1 over the wide list (sort groups with more
                # than SWAP_HYBRID_DENSE_MIN_ROWS valid rows); it writes the
                # same E4M3 rows and blocked scales the swap sub-tiles write
                # for the narrow groups, so GEMM2 sees one contiguous layout.
                gemm1_tactic = w._tactic(num_tokens)[1]
                if gemm1_tactic[0][0] != self.group_rows:
                    raise ValueError(
                        "mixed-tile GEMM1 tactic tile must match the "
                        f"{self.group_rows}-row sort groups, got {gemm1_tactic!r}"
                    )
                blockscaled_contiguous_gather_grouped_gemm_act_fusion(
                    a=x,
                    b=w1,
                    a_scale=x_sf,
                    b_scale=w1_sf,
                    alpha=b["w1_alpha"],
                    tile_idx_to_expert_idx=b["out_tile_idx_to_expert_idx"],
                    tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
                    token_id_mapping=b["out_permuted_idx_to_expanded_idx"],
                    num_non_exiting_tiles=b["swap_wide_count"],
                    tile_idx_to_row_group=b["swap_wide_list"],
                    out=b["gemm1_out"],
                    out_scale=b["gemm1_out_scale"],
                    c_dtype="float8_e4m3fn",
                    a_dtype="float8_e4m3fn",
                    b_dtype="float4_e2m1fn",
                    sf_dtype="float8_e8m0fnu",
                    sf_vec_size=32,
                    quantize_output=True,
                    topk=w.top_k,
                    mma_tiler_mn=gemm1_tactic[0],
                    cluster_shape_mn=gemm1_tactic[1],
                    enable_pdl=w.enable_pdl,
                    activation_type=w.activation_type.value,
                    situ_beta=self._beta,
                    situ_linear_beta=self._linear_beta,
                    weight_l2_hint=DENSE_WEIGHT_L2_HINT,
                    _prepared_launches=launches,
                )
                self._gemm1_dense = launches["gather"]
            fused_finalize = self.finalize and not self.two_stage
            if self.hybrid:
                # Dense contiguous grouped GEMM2 over the 128-row sort groups
                # with the bulk-reduce finalize into the zero-filled output.
                gemm2_tactic = w._tactic(num_tokens)[2]
                if gemm2_tactic[0][0] != self.group_rows:
                    raise ValueError(
                        "hybrid GEMM2 tactic tile must match the "
                        f"{self.group_rows}-row sort groups, got {gemm2_tactic!r}"
                    )
                blockscaled_contiguous_grouped_gemm_finalize_fusion(
                    a=b["gemm1_out"],
                    b=w2,
                    a_scale=b["gemm1_out_scale"],
                    b_scale=w2_sf,
                    alpha=b["w2_alpha"],
                    tile_idx_to_expert_idx=b["out_tile_idx_to_expert_idx"],
                    num_non_exiting_tiles=b["out_num_non_exiting_tiles"],
                    tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
                    permuted_idx_to_expanded_idx=b["out_permuted_idx_to_expanded_idx"],
                    token_final_scales=self._route_weights,
                    out=self.output,
                    a_dtype="float8_e4m3fn",
                    b_dtype="float4_e2m1fn",
                    sf_dtype="float8_e8m0fnu",
                    sf_vec_size=32,
                    out_dtype="bfloat16",
                    mma_tiler_mn=gemm2_tactic[0],
                    cluster_shape_mn=gemm2_tactic[1],
                    enable_pdl=w.enable_pdl,
                    use_fused_finalize=True,
                    weight_l2_hint=DENSE_WEIGHT_L2_HINT,
                    _prepared_launches=launches,
                )
                launches["swap_gemm2"] = launches["finalize"]
            else:
                self._prepare_swap_gemm2(
                    w, b, w2, w2_sf, num_tokens, fused_finalize, launches
                )
            self._gemm1, self._gemm1_args = launches["swap_gemm1"]
            self._gemm2, self._gemm2_args = launches["swap_gemm2"]
            if self.two_stage:
                self._finalize_rows = plan_finalize_rows(
                    self._partial_rows,
                    b["out_expanded_idx_to_permuted_idx"],
                    self._route_weights,
                    self.output,
                )

    def _prepare_swap_gemm2(
        self, w, b, w2, w2_sf, num_tokens, fused_finalize, launches
    ):
        with torch.cuda.device(self.device):
            # Short GEMM2 stages (4 K blocks) pipeline deeper and win 6-10 us
            # on the wide expert-parallel shard while every expert fits one
            # row group (T <= 256); the 384-wide single stage of a narrow
            # MoE-TP shard is faster from T = 512 up and for few hot experts.
            gemm2_k_blocks = None
            if (
                num_tokens <= SWAP_GEMM2_SHORT_STAGE_MAX_TOKENS
                and w.intermediate_shard > SWAP_TWO_STAGE_MAX_SHARD
                and not os.environ.get("SWAPAB_KBLOCKS2")
            ):
                gemm2_k_blocks = 4
            gemm2_m_group = None
            if (
                w.intermediate_shard <= SWAP_TWO_STAGE_MAX_SHARD
                and SWAP_TP_GEMM2_MGROUP > 1
                and num_tokens >= SWAP_TP_GEMM2_MGROUP_MIN_TOKENS
                and not os.environ.get("SWAPAB_MGROUP2")
            ):
                gemm2_m_group = SWAP_TP_GEMM2_MGROUP
                if not os.environ.get("SWAPAB_KBLOCKS2"):
                    gemm2_k_blocks = 4
            gemm2_lists = {"num_non_exiting_tiles": b["out_num_non_exiting_tiles"]}
            if self.mixed:
                # Every occupied n_tile-row sub-tile of the 128-row groups;
                # the row scales come blocked from the mixed GEMM1 tiles.
                # Grouped weight tiles keep 128-wide (4 K-block) stages, as
                # ``gemm2_k_blocks_per_stage`` does for ``swap_m_group`` > 1.
                if SWAP_MIXED_GEMM2_MGROUP > 1 and not os.environ.get(
                    "SWAPAB_KBLOCKS2"
                ):
                    gemm2_k_blocks = 4
                gemm2_m_group = SWAP_MIXED_GEMM2_MGROUP
                gemm2_lists = dict(
                    num_non_exiting_tiles=b["swap_all_count"],
                    tile_idx_to_row_group=b["swap_all_groups"],
                    group_rows=self.group_rows,
                    sf_blocked=not SWAP_MIXED_SF_PLAIN,
                )
            swapab_gemm2(
                w2=w2,
                w2_sf=w2_sf,
                act=b["gemm1_out"],
                act_sf=b["gemm1_out_scale"],
                out=self._partial_rows if self.two_stage else self.output,
                tile_idx_to_expert_idx=b["out_tile_idx_to_expert_idx"],
                tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
                alpha=b["w2_alpha"],
                permuted_idx_to_expanded_idx=b["out_permuted_idx_to_expanded_idx"],
                token_final_scales=self._route_weights if fused_finalize else None,
                top_k=w.top_k,
                finalize=fused_finalize,
                n_tile=self.n_tile,
                k_blocks_per_stage=gemm2_k_blocks,
                enable_pdl=w.enable_pdl,
                weight_l2_hint=w._swap_weight_l2_hint(num_tokens),
                _prepared_launches=launches,
                m_group=gemm2_m_group,
                **gemm2_lists,
            )

    def run(self) -> torch.Tensor:
        """Enqueue three (T <= 16), four (deferred), five (two-stage
        finalize), six (hybrid: sort, dispatch, swap and dense GEMM1
        tiles, GEMM2) or up to seven (mixed: + two-stage finalize) launches
        on the caller's stream."""
        with torch.cuda.device(self.device):
            stream_ptr = torch.cuda.current_stream().cuda_stream
            stream = cuda.CUstream(stream_ptr)
            self._route_preprocess.run(stream)
            if self._sort is not None:
                self._sort(*self._sort_args, stream_ptr)
            if self._dispatch is not None:
                self._dispatch(*self._dispatch_args, stream_ptr)
            if self._token_index is not None:
                self._token_index(*self._token_index_args, stream=stream)
            self._gemm1(*self._gemm1_args, stream=stream)
            if self._gemm1_dense is not None:
                compiled, args, kwargs = self._gemm1_dense
                compiled(*args, stream=stream, **kwargs)
            self._gemm2(*self._gemm2_args, stream=stream)
            if self._finalize_rows is not None:
                self._finalize_rows.run(stream)
        return self.output


class CuteDslMxfp4MoEWrapper:
    """MXFP4 runner with offline tactics and explicit caller-owned workspace.

    The wrapper is metadata only. ``get_workspace_size(T)`` may be called
    before any CUDA allocation. ``plan`` compiles and performs warmup
    execution using valid caller inputs; it must run outside CUDA Graph
    capture. Its returned plan is used both for prefill and decode.

    ``offline_tactics`` maps token-count upper bounds to W4A8 tactic tuples.
    The smallest covering bucket is selected from host shape metadata. No
    serving-time tuning or device-to-host routing inspection is performed.
    An omitted table uses the conservative existing Blackwell tactic.

    ``num_experts`` and ``intermediate_size`` are the global model values.
    The rank-local layout is explicit: ``parallel_layout`` (uniform EP or MoE
    TP) and/or the expert interval ``num_local_experts``/``local_expert_offset``
    (the expert-parallel form). Both forms may be given if they agree. The
    derived ``layout`` fixes the weight shapes ``plan`` accepts; shapes never
    select the mode. Every rank computes a partial output; reduce externally.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        num_local_experts: Optional[int] = None,
        local_expert_offset: Optional[int] = None,
        parallel_layout: Optional[Mxfp4MoEParallelLayout] = None,
        activation_type: ActivationType = ActivationType.Situ,
        quantization: str = "mxfp4_w4a8",
        enable_pdl: bool = False,
        offline_tactics: Optional[dict] = None,
        swapab_max_tokens: Optional[int] = None,
        swapab_n_tile: int = SWAP_ROW_TILE,
        swapab_tile_policy=None,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = ActivationType(activation_type)
        self.quantization = quantization
        self.enable_pdl = enable_pdl
        # Swap-AB decode path: token counts up to ``swapab_max_tokens`` (0
        # disables) run weights-as-M grouped GEMMs with ``swapab_n_tile``-row
        # expert groups; the fused route preprocess bounds it to T <= 16.
        if swapab_max_tokens is not None and swapab_max_tokens < 0:
            raise ValueError("swapab_max_tokens must be >= 0")
        if swapab_n_tile not in (8, 16, 32, 64, 128):
            raise ValueError("swapab_n_tile must be 8, 16, 32, 64 or 128")
        self.swapab_max_tokens = swapab_max_tokens
        self.swapab_n_tile = swapab_n_tile
        # (max_tokens, rows per expert group) buckets for T > 16; T <= 16 uses
        # ``swapab_n_tile``. Weights are re-streamed once per group, so the
        # group width grows with the expected rows per local expert.
        policy = swapab_tile_policy
        if policy is None:
            env_policy = os.environ.get("SWAPAB_TILE_POLICY")
            if env_policy:
                # "256:8,0:32" (0 = no upper bound)
                policy = tuple(
                    (int(t) if int(t) > 0 else (1 << 62), int(n))
                    for t, n in (item.split(":") for item in env_policy.split(","))
                )
        if policy is not None:
            policy = tuple((int(t), int(n)) for t, n in policy)
            if any(n not in (8, 16, 32, 64, 128) for _, n in policy) or any(
                policy[i][0] >= policy[i + 1][0] for i in range(len(policy) - 1)
            ):
                raise ValueError(
                    "swapab_tile_policy must be ascending "
                    "(max_tokens, tile in {8,16,32,64,128})"
                )
        self._swapab_tile_policy = policy
        supported = mxfp4_moe_capability(
            gpu_arch=103,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            parallel_layout=parallel_layout,
            quantization=quantization,
            activation_type=self.activation_type,
        )
        if not supported.supported:
            raise ValueError(supported.reason)
        self.parallel_layout = parallel_layout
        self.layout = supported.layout
        self.num_local_experts = self.layout.num_local_experts
        self.local_expert_offset = self.layout.local_expert_offset
        self.intermediate_shard = self.layout.intermediate_shard
        if self.swapab_max_tokens is None:
            env_max = os.environ.get("SWAPAB_MAX_TOKENS")
            if env_max:
                self.swapab_max_tokens = int(env_max)
            else:
                # Measured on B300 (Kimi K3 geometry, two-stage finalize): the
                # swap path beats the dense grouped GEMMs up to T=1024 in both
                # layouts; beyond that every expert needs a second 32-row
                # group (weights re-streamed) and wider groups pay more in
                # the row operand and finalize than they save. The narrow
                # MoE-TP shard continues with the hybrid form (64-row swap
                # GEMM1 sub-tiles + dense finalize GEMM2) up to T=2048; at
                # T=4096 the dense path is faster again (the 64-row GEMM1
                # re-streams every expert's weights for 2-3 groups).
                self.swapab_max_tokens = (
                    2048 if SWAP_HYBRID and self.intermediate_shard < 1024 else 1024
                )
        if self._swapab_tile_policy is None:
            # Measured on B300 (Kimi K3). The group width follows the local
            # rows per expert (T/56 under balanced routing) so that no expert
            # needs a second group, while a wider group caps the weight
            # re-streaming of a hot expert; the narrow MoE-TP shard has 8x
            # the local rows of an EP8 rank at the same T.
            if self.intermediate_shard >= 1024:
                self._swapab_tile_policy = ((128, 16), (1 << 62, 32))
            elif SWAP_HYBRID:
                self._swapab_tile_policy = (
                    (128, 8),
                    (256, 16),
                    (1024, 32),
                    (1 << 62, SWAP_HYBRID_MIN_TILE),
                )
            else:
                self._swapab_tile_policy = ((128, 8), (256, 16), (1 << 62, 32))
        self.swapab_tile_policy = self._swapab_tile_policy
        self._offline_tactics = sorted(
            (int(limit), canonicalize_w4a8_tactic(tactic))
            for limit, tactic in (offline_tactics or {}).items()
        )
        if any(limit <= 0 for limit, _ in self._offline_tactics):
            raise ValueError("offline tactic bucket bounds must be positive")

    @property
    def parallel_mode(self) -> str:
        return self.layout.mode

    def _metadata(self):
        return dict(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            parallel_layout=self.parallel_layout,
            quantization=self.quantization,
            activation_type=self.activation_type,
        )

    def _tactic(self, num_tokens):
        for limit, tactic in self._offline_tactics:
            if num_tokens <= limit:
                return tactic
        if self.activation_type == ActivationType.Situ:
            table = (
                B300_SITU_DENSE_TACTIC_TABLE_NARROW
                if self.intermediate_shard < 1024
                else B300_SITU_DENSE_TACTIC_TABLE_WIDE
            )
            for limit, tactic in table:
                if num_tokens <= limit:
                    return tactic
        return DEFAULT_BLACKWELL_MOE_TACTIC

    def _use_swapab(self, num_tokens, do_finalize=True):
        # Deferred output exists only on the swap-AB path, at any token count.
        return (
            (1 <= num_tokens <= self.swapab_max_tokens or not do_finalize)
            and self.activation_type == ActivationType.Situ
            and not self.enable_pdl
        )

    def _dense_two_stage(self, num_tokens):
        # Dense path: expanded-row GEMM2 output + finalize kernel instead of
        # the bulk reduce-add epilogue (experimental, env-gated).
        return DENSE_TWO_STAGE_FINALIZE and num_tokens > SWAP_ATOMIC_FINALIZE_MAX_TOKENS

    def _swap_weight_l2_hint(self, num_tokens):
        """L2 policy for the swap GEMMs' weight loads at this token count."""
        if num_tokens <= SWAP_WEIGHT_L2_HINT_MAX_TOKENS:
            return _DENSE_L2_HINTS["first"]
        lo, hi = SWAP_EP_L2HINT_SKIP
        if self.parallel_mode == "expert_parallel" and lo <= num_tokens <= hi:
            return None
        return _DENSE_L2_HINTS["first"]

    def _swap_tile(self, num_tokens):
        if num_tokens <= 16:
            return self.swapab_n_tile
        for max_tokens, tile in self.swapab_tile_policy:
            if num_tokens <= max_tokens:
                return tile
        return self.swapab_tile_policy[-1][1]

    def _swap_hybrid(self, num_tokens, do_finalize=True):
        """Hybrid form: swap GEMM1 sub-tiles of 128-row sort groups + dense
        finalize GEMM2 (wide swap tiles, finalize=True, generic routing)."""
        return (
            SWAP_HYBRID
            and bool(do_finalize)
            and num_tokens * self.top_k > FUSED_ROUTE_MAX_ROUTES
            and self._swap_tile(num_tokens) >= SWAP_HYBRID_MIN_TILE
        )

    def _swap_mixed(self, num_tokens, do_finalize=True):
        """Mixed form: 128-row sort groups with dense / swap GEMM1 tiles and
        the swap GEMM2 of the policy tile over every occupied sub-tile."""
        return (
            SWAP_MIXED
            and bool(do_finalize)
            and num_tokens >= SWAP_MIXED_MIN_TOKENS
            and (self.intermediate_shard < 1024 or SWAP_MIXED_EP)
            and not self._swap_hybrid(num_tokens, do_finalize)
            and SWAP_HYBRID_GROUP_ROWS % self._swap_tile(num_tokens) == 0
        )

    def _swap_group_rows(self, num_tokens, do_finalize=True):
        if self._swap_hybrid(num_tokens, do_finalize) or self._swap_mixed(
            num_tokens, do_finalize
        ):
            return SWAP_HYBRID_GROUP_ROWS
        return self._swap_tile(num_tokens)

    def _workspace_fields(self, num_tokens, do_finalize=True):
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self._use_swapab(num_tokens, do_finalize):
            tile = self._swap_tile(num_tokens)
            hybrid = self._swap_hybrid(num_tokens, do_finalize)
            mixed = self._swap_mixed(num_tokens, do_finalize)
            group = self._swap_group_rows(num_tokens, do_finalize)
            tiles = get_max_num_tiles(
                num_tokens, self.top_k, self.num_local_experts, group
            )
            rows = tiles * group
            specs = [
                ("out_tile_idx_to_expert_idx", (tiles,), torch.int32, 4),
                ("out_tile_idx_to_mn_limit", (tiles,), torch.int32, 4),
                *(
                    # Dispatch work lists of the hybrid form.
                    [
                        ("swap_row_groups", (tiles * (group // tile),), torch.int32, 4),
                        ("swap_row_group_count", (1,), torch.int32, 4),
                        ("swap_wide_list", (tiles,), torch.int32, 4),
                        ("swap_wide_count", (1,), torch.int32, 4),
                    ]
                    if (hybrid or mixed)
                    else []
                ),
                *(
                    # Mixed form: the swap GEMM2 work list (every occupied sub-tile).
                    [
                        ("swap_all_groups", (tiles * (group // tile),), torch.int32, 4),
                        ("swap_all_count", (1,), torch.int32, 4),
                    ]
                    if mixed
                    else []
                ),
                # moe_sort scratch (T > 16 path; used by the sort for T > 1024)
                ("out_expert_counts", (2 * 4096,), torch.int32, 4),
                (
                    "out_expanded_idx_to_permuted_idx",
                    (num_tokens, self.top_k),
                    torch.int32,
                    4,
                ),
                ("out_permuted_idx_to_expanded_idx", (rows,), torch.int32, 4),
                *(
                    # gather4 row coordinates of the TMA row operand
                    [("permuted_idx_to_token_idx", (rows,), torch.int32, 4)]
                    if swap_row_tma(tile, True)
                    else []
                ),
                ("out_total_num_padded_tokens", (1,), torch.int32, 4),
                ("out_num_non_exiting_tiles", (1,), torch.int32, 4),
                ("gemm1_out", (rows, self.intermediate_shard), torch.float8_e4m3fn, 1),
                (
                    "gemm1_out_scale",
                    (rows, self.intermediate_shard // 32),
                    torch.uint8,
                    1,
                ),
                ("route_ids", (num_tokens, self.top_k), torch.int32, 4),
                ("route_weights", (num_tokens, self.top_k), torch.float32, 4),
                ("w1_alpha", (self.num_local_experts,), torch.float32, 4),
                ("w2_alpha", (self.num_local_experts,), torch.float32, 4),
            ]
            if (
                do_finalize
                and not hybrid
                and num_tokens > SWAP_ATOMIC_FINALIZE_MAX_TOKENS
                and self.intermediate_shard <= SWAP_TWO_STAGE_MAX_SHARD
            ):
                # Two-stage finalize: GEMM2's alpha-scaled rows in permuted
                # order, reduced by the finalize kernel.
                specs.append(
                    ("partial_rows", (rows, self.hidden_size), torch.bfloat16, 2)
                )
            fields, offset = [], 0
            for name, shape, dtype, itemsize in specs:
                offset = _align(offset)
                size = prod(shape) * itemsize
                fields.append(_WorkspaceField(name, shape, dtype, offset, size))
                offset += size
            return fields, _align(offset)
        tile = self._tactic(num_tokens)[0]
        tiles = get_max_num_tiles(num_tokens, self.top_k, self.num_local_experts, tile)
        rows = tiles * tile
        specs = [
            *(
                [
                    (
                        "partial_rows",
                        (num_tokens * self.top_k, self.hidden_size),
                        torch.bfloat16,
                        2,
                    )
                ]
                if self._dense_two_stage(num_tokens)
                else []
            ),
            ("out_tile_idx_to_expert_idx", (tiles,), torch.int32, 4),
            ("out_tile_idx_to_mn_limit", (tiles,), torch.int32, 4),
            (
                "out_expanded_idx_to_permuted_idx",
                (num_tokens, self.top_k),
                torch.int32,
                4,
            ),
            ("out_permuted_idx_to_expanded_idx", (rows,), torch.int32, 4),
            ("out_total_num_padded_tokens", (1,), torch.int32, 4),
            ("out_num_non_exiting_tiles", (1,), torch.int32, 4),
            ("gemm1_out", (rows, self.intermediate_shard), torch.float8_e4m3fn, 1),
            (
                "gemm1_out_scale",
                (32, 4, rows // 128, 4, self.intermediate_shard // 128, 1),
                torch.uint8,
                1,
            ),
            ("route_ids", (num_tokens, self.top_k), torch.int32, 4),
            ("route_weights", (num_tokens, self.top_k), torch.float32, 4),
            ("w1_alpha", (self.num_local_experts,), torch.float32, 4),
            ("w2_alpha", (self.num_local_experts,), torch.float32, 4),
        ]
        if num_tokens > 1024:
            specs.append(("out_expert_counts", (2 * self.num_experts,), torch.int32, 4))
        fields, offset = [], 0
        for name, shape, dtype, itemsize in specs:
            offset = _align(offset)
            size = prod(shape) * itemsize
            fields.append(_WorkspaceField(name, shape, dtype, offset, size))
            offset += size
        return fields, _align(offset)

    def get_workspace_size(self, num_tokens: int, do_finalize: bool = True) -> int:
        """Return required workspace bytes for any routing at this token count.

        The size follows the rank-local layout: the GEMM1 intermediate region
        uses ``intermediate_shard`` columns and the per-expert regions use
        ``num_local_experts``. ``do_finalize=False`` sizes the deferred-output
        (swap-AB) plan for this token count.
        """
        return self._workspace_fields(num_tokens, do_finalize)[1]

    def get_deferred_output_rows(self, num_tokens: int) -> int:
        """Rows of the caller-owned ``[rows, hidden_size]`` BF16 buffer that
        ``plan(..., do_finalize=False)`` writes: the permuted-row capacity
        (every local expert's rows padded to the row-group width) for any
        routing at this token count."""
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if not self._use_swapab(num_tokens, do_finalize=False):
            raise ValueError(
                "deferred output requires SiTU activation without PDL "
                "(the swap-AB path)"
            )
        tile = self._swap_tile(num_tokens)
        return (
            get_max_num_tiles(num_tokens, self.top_k, self.num_local_experts, tile)
            * tile
        )

    def plan(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        w1: torch.Tensor,
        w1_sf: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        *,
        beta: Optional[torch.Tensor] = None,
        linear_beta: Optional[torch.Tensor] = None,
        workspace: torch.Tensor,
        output: torch.Tensor,
        do_finalize: bool = True,
    ) -> Union[Mxfp4MoEPlan, Mxfp4MoESwapAbPlan]:
        """Bind buffers and prepare kernels; all tensor contents must be valid.

        Weights use ``prepare_cute_dsl_mxfp4_weights`` layouts for this rank's
        shard: ``[num_local_experts, 2*intermediate_shard, H/2]`` W1 and
        ``[num_local_experts, H, intermediate_shard/2]`` W2, as produced by
        ``shard_cute_dsl_mxfp4_weights`` for the resolved layout. Shapes are
        validated against that layout and never used to select it. ``x_sf`` is
        linear UE8M0 bytes [T,H/32]. ``beta`` and optional ``linear_beta`` are
        contiguous CUDA FP32 tensors with one value or one per local expert.
        Their values must be finite and positive; they are read on the device
        at execution, so changing them requires no recompilation.

        Routing may be separate int32 IDs and BF16/FP32 weights, or packed
        int32 (expert ID in high 16 bits, BF16 weight in low 16 bits) when
        ``topk_weights=None``. IDs are global, must be in ``[0, num_experts)``,
        and must be distinct within each token. Output and workspace must be
        distinct from all inputs; workspace is a contiguous uint8 tensor whose
        address is aligned to 256 bytes.

        ``do_finalize=False`` (deferred finalize, SiTU only) skips the
        route-weight reduction: ``output`` is a caller-owned contiguous BF16
        ``[rows, H]`` buffer with ``rows >= get_deferred_output_rows(T)``
        that receives ``alpha * GEMM2`` rows in permuted order, and the plan
        exposes ``expanded_idx_to_permuted_idx`` (int32 ``[T, top_k]``, ``-1``
        for non-local routes) and ``route_weights`` (FP32 ``[T, top_k]``),
        both valid after ``run``. Rows not referenced by the map are padding.
        """
        if x.device.type != "cuda":
            raise ValueError("plan requires CUDA tensors")
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("plan must be called before CUDA Graph capture")
            major, minor = torch.cuda.get_device_capability(x.device)
            capability = mxfp4_moe_capability(
                gpu_arch=major * 10 + minor, do_finalize=do_finalize, **self._metadata()
            )
            if not capability.supported:
                raise ValueError(capability.reason)
            num_tokens = x.shape[0]
            layout = self.layout
            if not do_finalize:
                if not self._use_swapab(num_tokens, do_finalize=False):
                    raise ValueError(
                        "deferred output requires SiTU activation without PDL"
                    )
                deferred_rows = self.get_deferred_output_rows(num_tokens)
                if (
                    output.device != x.device
                    or output.dtype != torch.bfloat16
                    or output.ndim != 2
                    or output.shape[0] < deferred_rows
                    or output.shape[1] != self.hidden_size
                    or not output.is_contiguous()
                ):
                    raise ValueError(
                        "deferred output must be contiguous BF16 [>= "
                        f"{deferred_rows}, {self.hidden_size}] on {x.device}; got "
                        f"{output.dtype} {tuple(output.shape)} on {output.device}"
                    )
            expected = {
                "x": (x, (num_tokens, self.hidden_size), torch.float8_e4m3fn),
                "x_sf": (x_sf, (num_tokens, self.hidden_size // 32), torch.uint8),
                "topk_ids": (topk_ids, (num_tokens, self.top_k), torch.int32),
                "w1": (
                    w1,
                    (
                        layout.num_local_experts,
                        2 * layout.intermediate_shard,
                        self.hidden_size // 2,
                    ),
                    torch.uint8,
                ),
                "w2": (
                    w2,
                    (
                        layout.num_local_experts,
                        self.hidden_size,
                        layout.intermediate_shard // 2,
                    ),
                    torch.uint8,
                ),
            }
            if do_finalize:
                expected["output"] = (
                    output,
                    (num_tokens, self.hidden_size),
                    torch.bfloat16,
                )
            for name, (tensor, shape, dtype) in expected.items():
                if (
                    tensor.device != x.device
                    or tensor.dtype != dtype
                    or tuple(tensor.shape) != shape
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} must be contiguous {dtype} {shape} on {x.device} "
                        f"for parallel mode {layout.mode} ({layout.num_local_experts} "
                        f"local experts at offset {layout.local_expert_offset}, "
                        f"intermediate shard {layout.intermediate_shard}); got "
                        f"{tensor.dtype} {tuple(tensor.shape)} on {tensor.device}"
                    )
            for name, tensor, rows, columns in (
                ("w1_sf", w1_sf, 2 * layout.intermediate_shard, self.hidden_size),
                ("w2_sf", w2_sf, self.hidden_size, layout.intermediate_shard),
            ):
                expected_shape = (
                    32,
                    4,
                    rows // 128,
                    4,
                    columns // 128,
                    layout.num_local_experts,
                )
                if (
                    tensor.device != x.device
                    or tensor.dtype != torch.uint8
                    or tuple(tensor.shape) != expected_shape
                ):
                    raise ValueError(
                        f"{name} must be the prepared uint8 MMA scale layout "
                        f"{expected_shape} on {x.device} for parallel mode "
                        f"{layout.mode}; got {tensor.dtype} {tuple(tensor.shape)} "
                        f"on {tensor.device}"
                    )
            if topk_weights is not None and (
                topk_weights.device != x.device
                or topk_weights.dtype not in (torch.bfloat16, torch.float32)
                or tuple(topk_weights.shape) != (num_tokens, self.top_k)
                or not topk_weights.is_contiguous()
            ):
                raise ValueError(
                    "topk_weights must be contiguous CUDA BF16/FP32 [T,top_k]"
                )
            if self.activation_type == ActivationType.Situ and beta is None:
                raise ValueError("SiTU requires runtime beta")
            if self.activation_type != ActivationType.Situ and (
                beta is not None or linear_beta is not None
            ):
                raise ValueError("SiTU parameters require ActivationType.Situ")
            for name, tensor in (("beta", beta), ("linear_beta", linear_beta)):
                if tensor is not None and (
                    tensor.device != x.device
                    or tensor.dtype != torch.float32
                    or tensor.ndim != 1
                    or tensor.numel() not in (1, self.num_local_experts)
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} must be CUDA FP32 [1] or [num_local_experts]"
                    )
            fields, size = self._workspace_fields(num_tokens, do_finalize)
            if (
                workspace.device != x.device
                or workspace.dtype != torch.uint8
                or workspace.ndim != 1
                or not workspace.is_contiguous()
                or workspace.numel() < size
                or workspace.data_ptr() % 256
            ):
                raise ValueError(
                    f"workspace requires at least {size} aligned CUDA uint8 bytes"
                )
            workspace_interval = (workspace.data_ptr(), workspace.data_ptr() + size)
            output_interval = _byte_interval(output)
            if _overlap(workspace_interval, output_interval):
                raise ValueError("output must not overlap workspace")
            for name, tensor in (
                ("x", x),
                ("x_sf", x_sf),
                ("topk_ids", topk_ids),
                ("topk_weights", topk_weights),
                ("w1", w1),
                ("w1_sf", w1_sf),
                ("w2", w2),
                ("w2_sf", w2_sf),
                ("beta", beta),
                ("linear_beta", linear_beta),
            ):
                if tensor is not None and (
                    _overlap(workspace_interval, _byte_interval(tensor))
                    or _overlap(output_interval, _byte_interval(tensor))
                ):
                    raise ValueError(f"output/workspace must not overlap {name}")
            buffers = {
                f.name: workspace.narrow(0, f.offset, f.nbytes)
                .view(f.dtype)
                .view(f.shape)
                for f in fields
            }
            buffers["w1_alpha"].fill_(1.0)
            buffers["w2_alpha"].fill_(1.0)
            route_weights = (
                topk_weights
                if topk_weights is not None and topk_weights.dtype == torch.float32
                else buffers["route_weights"]
            )
            validate_w4a8_inputs(x, x_sf, route_weights, w1, w1_sf, w2, w2_sf)
            if w1_sf.device != x.device or w2_sf.device != x.device:
                raise ValueError("weight scales must be on the input device")
            if self._use_swapab(num_tokens, do_finalize):
                if (major, minor) not in ((10, 0), (10, 3)):
                    raise ValueError("the swap-AB decode path requires SM100/SM103")
                swap_plan = Mxfp4MoESwapAbPlan(
                    wrapper=self,
                    buffers=buffers,
                    workspace=workspace,
                    x=x,
                    x_sf=x_sf,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    w1=w1,
                    w1_sf=w1_sf,
                    w2=w2,
                    w2_sf=w2_sf,
                    beta=beta,
                    linear_beta=linear_beta,
                    output=output,
                    n_tile=self._swap_tile(num_tokens),
                    finalize=do_finalize,
                )
                swap_plan._prepare()
                return swap_plan
            tile, gemm1, gemm2 = self._tactic(num_tokens)
            # The public routing contract requires distinct IDs per token,
            # so an expert has at most T rows. Restrict this specialization
            # to the qualified B300 SiTU decode tactic.
            decode_specialization = (
                (major, minor) == (10, 3)
                and 1 <= num_tokens <= 16
                and self.activation_type == ActivationType.Situ
                and not self.enable_pdl
                and tile == 128
                and gemm1 == ((128, 128), (1, 1), False)
                and gemm2 == ((128, 128), (1, 1), False)
            )
            route_ids = buffers["route_ids"] if topk_weights is None else topk_ids
            dense_two_stage = self._dense_two_stage(num_tokens)
            kwargs = dict(
                x=x,
                x_sf=x_sf,
                token_selected_experts=route_ids,
                token_final_scales=route_weights,
                w1_weight=w1,
                w1_weight_sf=w1_sf,
                w1_alpha=buffers["w1_alpha"],
                fc2_input_scale=None,
                w2_weight=w2,
                w2_weight_sf=w2_sf,
                w2_alpha=buffers["w2_alpha"],
                num_experts=self.num_experts,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_offset,
                tile_size=tile,
                gemm1_mma_tiler_mn=gemm1[0],
                gemm1_cluster_shape_mn=gemm1[1],
                gemm2_mma_tiler_mn=gemm2[0],
                gemm2_cluster_shape_mn=gemm2[1],
                moe_sort_buffers={
                    name: value
                    for name, value in buffers.items()
                    if name.startswith("out_")
                },
                gemm1_out=buffers["gemm1_out"],
                gemm1_out_scale=buffers["gemm1_out_scale"],
                moe_output=output,
                output_dtype=torch.bfloat16,
                use_async_memset=False,
                use_fused_finalize=not dense_two_stage,
                gemm2_partial_out=buffers["partial_rows"] if dense_two_stage else None,
                skip_unpermute=dense_two_stage,
                weight_l2_hint=DENSE_WEIGHT_L2_HINT,
                enable_pdl=self.enable_pdl,
                activation_type=self.activation_type.value,
                situ_beta=beta,
                situ_linear_beta=linear_beta,
                _enable_decode_specialization=decode_specialization,
            )
            plan = Mxfp4MoEPlan(
                kwargs=kwargs,
                workspace=workspace,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                route_ids=route_ids,
                route_weights=route_weights,
            )
            plan._prepare()
            return plan
