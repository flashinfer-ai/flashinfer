# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sage attention scale addressing for the decode softmax and epilogue.

Every scale load of the kernel goes through this module. The softmax bodies
load ``sfQ`` of their Q row once per work tile and the raw ``sfK`` of each
KV tile's scale groups; the epilogue receives the per-channel V scales for one
output column range.

Both providers index the trtllm-gen flat layout through ``flat_scale_slot``
of :mod:`flashinfer.attention.prims_ts.sage`, the module that owns the layout.
Tokens beyond the sequence end (masked score columns) and Q rows beyond the
valid row count are clamped to the last valid slot: their scores are masked or
discarded, but their scale must stay finite so masked columns exponentiate to
zero.

The contiguous provider derives each fragment's first token from the tile
offset. The block-sparse provider derives it from the route's K64 atom
origins; a proxy route reads ``k_summary_scale`` instead of ``k_scale`` and
indexes the summary sequence (one summary per KV block) with the same
arithmetic, which is why the K source is passed as an address, a head stride
and a sequence length.

The ``sfK`` words of one softmax instance are a task-scheduling resource,
:class:`SageKScalesResource`, defined below.

Sage attention scales every score by the K scale of its token block. A
softmax instance takes one KV tile's ``sfK`` words ahead of the tile's score
wait so the loads hide behind the QK MMA, and both softmax passes (the max
pass and the P pass) read them one K32 fragment at a time.

:class:`SageKScalesResource` owns where those words live and how they get
there. The softmax task produces and consumes it itself: ``publish_tile``
(producer work) fills the tile's words into the storage the resource owns,
``take_tile`` (consumer work) hands the passes the routed register array, and
the passes read fragments through ``open`` / ``fragment`` / ``advance``.

Two storage forms exist, selected at compile time by the scale groups per K32
fragment (``FmhaDecodeConfig.sage_k_scales_in_smem_for``):

* Register form (K blocks of 16 tokens and larger, at most two groups per
  fragment): each lane takes the words of its own fragments into a rotating
  register array. Registers are private to the lane, so the resource carries
  no pipeline and no producer step; ``take_tile`` gathers the words and
  routes the array.
* SMEM form (K blocks of 4 and 1 token, eight or 32 groups per fragment):
  the tile's words live in an SMEM ring of the instance, one slot per
  pipeline stage, in the layout of ``sage_k_scale_words``. The resource is a
  two-stage pipeline whose producer and consumer are both the instance's
  warps: the task acquires a slot, ``publish_tile`` fills it with the
  instance's threads, the commit publishes it, the wait makes it visible to
  both passes, which read one fragment's groups with 16-byte broadcast loads,
  and the release after the P pass frees the slot for the tile after next.
  The routed array is a one-word placeholder in this form.

The words come from one of three sources, chosen by the plan: a dense tile
gathers ``k_scale`` over its token range; a block-sparse route takes the
words the load warp staged with the route metadata; a proxy route of a plan
whose summary scales use another K block size (``sage_mixed_k_geometry``)
gathers ``k_summary_scale`` from the route's staged atom origins. A mixed
plan holds two resources per instance, one per geometry, each routing its own
array; the route kind, uniform across the CTA, selects which one serves the
tile, and the other one leaves its array at ones.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ....sage import flat_scale_slot, log2_block_size
from ...placeholder_helpers import _placeholder_local_array
from ..fmha_decode_config import FmhaDecodeConfig
from .helpers_common import (
    DecodeGenResourceBase,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _decode_gen_task_cache,
    _keeps_route_atom,
    _keeps_spatial_half,
    _logical_head_batch,
    _route_is_proxy,
    fmul2,
)
from .helpers_kv_tile_idx import resolve_keeps_tile_context

if TYPE_CHECKING:
    from .smem_block_sparse_metadata import SmemBlockSparseSoftmaxMetadataResource

Constexpr = cutlass.Constexpr


@dataclass(eq=False)
class SageScaleTensors:
    """The plan's Sage scale tensors as the kernel receives them.

    A plain mutable dataclass: the DSL swaps frozen dataclasses for proxies
    while it traces a dynamic branch, which changes the traced structure of
    the resource that holds one. Every resource that reads a scale takes
    this one object instead of its own copy of the pointers: ``sfQ`` and ``sfK`` are ``[Hkv, slots]`` FP32
    arrays in the flat layout with a per-head stride, ``k_summary_scale`` is
    the same for the summary sequence of a proxy plan (``None`` otherwise),
    and the per-channel V scales and means are ``[Hkv, D]`` FP32 (``v_mean``
    is ``None`` unless the recipe restores a channel mean).
    """

    q_scale_ptr: cute.Pointer
    q_scale_head_stride: Int32
    k_scale_ptr: cute.Pointer
    k_scale_head_stride: Int32
    k_summary_scale_ptr: cute.Pointer | None
    k_summary_scale_head_stride: Int32 | None
    v_scale_ptr: cute.Pointer
    v_mean_ptr: cute.Pointer | None


@cute.jit
def _load_scale(scale_addr: Int64, index: Int32) -> Float32:
    """Return one FP32 scale loaded from a global memory base address."""
    value_ptr = cutlass.inttoptr(
        scale_addr + Int64(index) * Int64(4),
        mem_space=1,
        dtype=Float32,
    )
    return Float32(value_ptr.load(count=1, alignment=4)[0])


@cute.jit
def load_q_scale(
    cfg: Constexpr[FmhaDecodeConfig],
    q_scale_addr: Int64,
    q_scale_head_stride: Int32,
    *,
    kv_head_idx: Int32,
    local_head_idx: Int32,
    batch_idx: Int32,
    q_token_idx: Int32,
) -> Float32:
    """Return ``sfQ`` for one Q row (token and Q head of one KV head group)."""
    heads_q_per_kv = Int32(cfg.heads_q_per_kv)
    q_head_idx = kv_head_idx * heads_q_per_kv + cute.math.min(
        local_head_idx, heads_q_per_kv - Int32(1)
    )
    q_token_idx = cute.math.min(q_token_idx, Int32(cfg.max_seq_len_q - 1))
    slot = flat_scale_slot(
        batch_idx,
        q_token_idx,
        Int32(cfg.max_seq_len_q),
        log2_block_size(cfg.sage_q_block_size),
    )
    return _load_scale(q_scale_addr, q_head_idx * q_scale_head_stride + slot)


@cute.jit
def load_k_scale(
    cfg: Constexpr[FmhaDecodeConfig],
    k_scale_addr: Int64,
    k_scale_head_stride: Int32,
    *,
    kv_head_idx: Int32,
    batch_idx: Int32,
    seq_len_kv: Int32,
    kv_token_idx: Int32,
    log2_k_block_size: Int32 | None = None,
) -> Float32:
    """Return ``sfK`` for one token of one KV head from a flat scale array.

    ``k_scale_addr`` is the base address of ``k_scale`` or ``k_summary_scale``,
    ``seq_len_kv`` the length of the sequence it covers and
    ``log2_k_block_size`` the log2 of its K block size, the recipe's token
    block by default.
    """
    log2_block = log2_k_block_size
    if cutlass.const_expr(log2_block is None):
        log2_block = Int32(log2_block_size(cfg.sage_k_block_size))
    kv_token_idx = cute.math.min(kv_token_idx, seq_len_kv - Int32(1))
    slot = flat_scale_slot(batch_idx, kv_token_idx, seq_len_kv, log2_block)
    return _load_scale(k_scale_addr, kv_head_idx * k_scale_head_stride + slot)


def sage_scale_arr_size(cfg: FmhaDecodeConfig, groups: int) -> int:
    """Return the number of ``sfK`` words one softmax lane holds per tile.

    ``groups`` is the scale groups per fragment of the tile's route kind
    (``FmhaDecodeConfig.sage_k_groups_per_fragment_for``).
    """
    return cfg.num_softmax_score_fragments * groups


def sage_k_scale_words(cfg: FmhaDecodeConfig, groups: int) -> int:
    """Return the number of ``sfK`` words that cover one KV tile for every spatial half.

    Zero without Sage attention, so the block-sparse staging layout can take
    the count unconditionally. Word ``half * arr_size + f * groups + g`` holds
    the scale of group ``g`` of fragment ``f`` of the lanes in that half
    (``sage_scale_arr_size`` words per half); this is the layout of the words
    a block-sparse route stages and of the SMEM ring of
    :class:`SageKScalesResource`, so a softmax thread reads its half's values with
    contiguous vector loads. ``groups`` selects the route kind's geometry.
    """
    if not cfg.use_sage_attention:
        return 0
    return cfg.keeps_spatial_halves * sage_scale_arr_size(cfg, groups)


def sage_staged_k_scale_words(cfg: FmhaDecodeConfig) -> int:
    """Return the ``sfK`` words a block-sparse route stages.

    Exact routes stage their words; proxy routes of a mixed-geometry plan
    gather theirs in the softmax warps, so the staged area covers the exact
    geometry alone.
    """
    return sage_k_scale_words(cfg, cfg.sage_k_groups_per_fragment)


@cute.jit
def sage_word_position(
    cfg: Constexpr[FmhaDecodeConfig],
    half: Int32,
    lane_entry,
    groups: Constexpr[int],
) -> tuple[Int32, Int32]:
    """Return ``(atom, token offset in the atom)`` of one ``sfK`` word.

    The word is entry ``lane_entry = f * groups + g`` of spatial half
    ``half``: group ``g`` of fragment ``f`` of the half's lane array. Fragment
    ``f`` reads the half's atom number ``f // fragments_per_atom``
    (``_keeps_route_atom``) from token ``(f % fragments_per_atom) *
    fragment_regs`` onward, and group ``g`` starts ``g * group_tokens`` later.
    The atom is the Keeps layout atom
    (``FmhaDecodeConfig.keeps_fragments_per_atom``), which block-sparse routes
    share as their route atom. A constant ``lane_entry`` folds everything but
    the half interleave at trace time. ``groups`` is the route kind's scale
    groups per fragment.
    """
    fragments_per_atom = cfg.keeps_fragments_per_atom
    fragment_regs = cfg.softmax_score_fragment_regs
    group_tokens = fragment_regs // groups
    fragment_idx = lane_entry // groups
    group_idx = lane_entry % groups
    atom_idx = _keeps_route_atom(cfg, half, fragment_idx // fragments_per_atom)
    token_offset = (
        fragment_idx % fragments_per_atom
    ) * fragment_regs + group_idx * group_tokens
    return atom_idx, Int32(token_offset)


@cute.jit
def dense_k_scale_token(
    cfg: Constexpr[FmhaDecodeConfig], half: Int32, lane_entry, tile_offset_k: Int32
) -> Int32:
    """Return the first KV token covered by one ``sfK`` word of a dense tile.

    A dense Keeps tile is its layout atoms in order (``_keeps_score_col``), so
    the word's position (``sage_word_position``) in a route of layout atoms
    names the token.
    """
    atom_idx, token_offset = sage_word_position(
        cfg, half, lane_entry, cfg.sage_k_groups_per_fragment
    )
    return tile_offset_k + atom_idx * Int32(cfg.keeps_atom_tokens) + token_offset


@cute.jit
def scale_pairs_in_place(
    values: cutlass.Array, factor: Float32, count: Constexpr[int]
) -> None:
    """Multiply ``count`` values in place by ``factor``, two at a time."""
    assert count % 2 == 0
    for pair_base in cutlass.range_constexpr(0, count, 2):
        values[pair_base], values[pair_base + 1] = fmul2(
            (factor, factor),
            (Float32(values[pair_base]), Float32(values[pair_base + 1])),
        )


@cute.jit
def block_sparse_k_scale_source(
    cfg: Constexpr[FmhaDecodeConfig],
    scales: SageScaleTensors,
    *,
    seq_len_kv: Int32,
    route_is_proxy: cutlass.Boolean,
) -> tuple[Int64, Int32, Int32, Int32]:
    """Return the K scale array of one block-sparse route.

    Exact routes read ``k_scale`` over the KV tokens with the recipe's K block
    size; proxy routes read ``k_summary_scale`` over the summary sequence,
    whose length is the number of KV blocks, with the summary K block size.
    The result is ``(base address, head stride, sequence length, log2 block
    size)`` for :func:`load_k_scale`.
    """
    scale_addr = scales.k_scale_ptr.toint()
    head_stride = scales.k_scale_head_stride
    seq_len = seq_len_kv
    log2_block = Int32(log2_block_size(cfg.sage_k_block_size))
    if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
        if route_is_proxy:
            scale_addr = scales.k_summary_scale_ptr.toint()
            head_stride = scales.k_summary_scale_head_stride
            seq_len = Int32(cfg.num_proxy_summaries)
            log2_block = Int32(log2_block_size(cfg.sage_k_summary_block_size))
    return scale_addr, head_stride, seq_len, log2_block


def staged_v_channel_scale_entries(cfg: FmhaDecodeConfig) -> int:
    """Return the number of SMEM floats holding one KV head's V scales and means.

    The scales occupy ``[0, headdim)``; with ``sage_v_mean`` the means follow
    at ``[headdim, 2 * headdim)``.
    """
    return cfg.headdim * (2 if cfg.sage_v_mean else 1)


@cute.jit
def stage_v_channel_scales(
    cfg: Constexpr[FmhaDecodeConfig],
    scales: SageScaleTensors,
    staged: cutlass.Array,
    *,
    kv_head_idx: Int32,
    thread_idx: Int32,
    num_threads: Constexpr[int],
) -> None:
    """Copy one KV head's per-channel V scales and means into SMEM.

    Both tensors are ``[Hkv, D]`` FP32. The ``num_threads`` callers each
    move ``headdim / num_threads`` channels, so the global loads are issued
    once per tile while the correction warps are otherwise idle, instead of
    inside the output store loop where their latency lands on the tile tail.
    Callers order the writes before the epilogue reads with a barrier, and
    place a second barrier after the last read of a tile so the next tile's
    writes cannot overtake it.
    """
    assert cfg.headdim % num_threads == 0
    v_scale_addr = scales.v_scale_ptr.toint()
    head_base = kv_head_idx * Int32(cfg.headdim)
    for base in cutlass.range_constexpr(0, cfg.headdim, num_threads):
        channel = Int32(base) + thread_idx
        staged[channel] = _load_scale(v_scale_addr, head_base + channel)
        if cutlass.const_expr(cfg.sage_v_mean):
            staged[channel + Int32(cfg.headdim)] = _load_scale(
                scales.v_mean_ptr.toint(), head_base + channel
            )


@cute.jit
def load_staged_v_channel_scales(
    cfg: Constexpr[FmhaDecodeConfig],
    staged: cutlass.Array,
    *,
    first_col: Int32,
    count: Constexpr[int],
) -> tuple[cutlass.Array, cutlass.Array]:
    """Return ``count`` staged V scales and means from ``first_col``.

    Without ``sage_v_mean`` the returned mean array is zero so callers can
    apply one fused multiply-add.
    """
    assert count % 4 == 0
    scales = cutlass.Array(Float32, count, space=cutlass.AddressSpace.rmem)
    means = cutlass.Array(Float32, count, space=cutlass.AddressSpace.rmem)
    for chunk in cutlass.range_constexpr(0, count, 4):
        chunk_col = first_col + Int32(chunk)
        scale_vec = (staged.data_ptr() + chunk_col).load(count=4, alignment=16)
        for elem in cutlass.range_constexpr(4):
            scales[chunk + elem] = Float32(scale_vec[elem])
        if cutlass.const_expr(cfg.sage_v_mean):
            mean_vec = (staged.data_ptr() + chunk_col + Int32(cfg.headdim)).load(
                count=4, alignment=16
            )
            for elem in cutlass.range_constexpr(4):
                means[chunk + elem] = Float32(mean_vec[elem])
        else:
            for elem in cutlass.range_constexpr(4):
                means[chunk + elem] = Float32(0.0)
    return scales, means


WordSource = Callable[[Int32, Int32, object], Float32]

SAGE_K_SCALES_RING_STAGES = 2


@dataclass(kw_only=True)
class SageKScalesResource(DecodeGenResourceBase):
    """One softmax instance's ``sfK`` words for one scale-group geometry.

    ``summary`` marks the resource of a mixed-geometry plan that serves proxy
    routes from ``k_summary_scale`` with the summary K block size; otherwise
    the resource serves every route with the token K block size. The SMEM
    form takes the instance's two-stage pipeline as ``pipeline_config``; the
    register form takes none. ``route_metadata`` is the instance's
    block-sparse softmax metadata resource, whose held stage holds the staged
    words and atom origins of the current route; a dense plan resolves the
    tile's token range from the sequence-length sources it shares with the S
    resource.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "sage_scale_arr",
            cutlass.Array,
            None,
            "Raw sfK words of the lane's fragments for the current tile.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    inst_id: Constexpr[int] = 0
    summary: Constexpr[bool] = False
    route_metadata: "SmemBlockSparseSoftmaxMetadataResource | None" = None
    seqlens_kv: cute.Pointer | None = None
    max_seq_len_kv: Int32 = None
    seq_len_q: Int32 = None
    q_group_idx: Int32 | None = None
    h_k_idx: Int32 | None = None
    b_idx: Int32 | None = None
    scale_tensors: SageScaleTensors | None = None
    sage_scale_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    _alloc: Constexpr[SmemAllocation | None] = None

    def __post_init__(self) -> None:
        assert (self.pipeline_config is not None) == self.in_smem, (
            "the SMEM form of the Sage K scales carries the instance's pipeline "
            "and the register form carries none"
        )
        super().__post_init__()

    @property
    def groups(self) -> int:
        """Return the scale groups per K32 fragment of the served geometry."""
        if self.summary:
            return self.cfg.sage_summary_k_groups_per_fragment
        return self.cfg.sage_k_groups_per_fragment

    @property
    def in_smem(self) -> bool:
        """Whether the tile's words live in the SMEM ring rather than registers."""
        return self.cfg.sage_k_scales_in_smem_for(self.groups)

    @property
    def arr_size(self) -> int:
        """Return the ``sfK`` words one lane holds per tile in the register form."""
        return sage_scale_arr_size(self.cfg, self.groups)

    @property
    def tile_words(self) -> int:
        """Return the ``sfK`` words of one tile slot of the SMEM ring."""
        return sage_k_scale_words(self.cfg, self.groups)

    @property
    def routed_words(self) -> int:
        """Return the length of the routed array: the lane's words, or one placeholder."""
        return 1 if self.in_smem else self.arr_size

    def _init_placeholder_state(self) -> None:
        """Create the placeholder routed array for task-graph tracing."""
        self.sage_scale_arr.default = _placeholder_local_array(
            Float32, self.routed_words
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the ring of the SMEM form; the register form needs none."""
        if not self.in_smem:
            return []
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=self.name,
                size_bytes=SAGE_K_SCALES_RING_STAGES * self.tile_words * 4,
                alignment=16,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """The words live in SMEM or registers only."""
        return []

    # -- schedule steps ----------------------------------------------------

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def publish_tile(self, stage_info: StageInfo) -> None:
        """Fill the acquired ring slot with a dense tile's words (SMEM form)."""
        self._fill_ring(stage_info, cutlass.Boolean(False))

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def publish_route_tile(self, stage_info: StageInfo, *, route_flags: Int32) -> None:
        """Fill the acquired ring slot with a block-sparse route's words (SMEM form).

        ``route_flags`` comes from the preceding ``load_route`` of the route
        metadata resource, whose stage must stay held until this step and
        ``take_route_tile`` have run.
        """
        self._fill_ring(stage_info, self._route_is_proxy(route_flags))

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=sage_scale_arr)
    @cute.jit
    def take_tile(self, stage_info: StageInfo) -> cutlass.Array:
        """Return the routed array of a dense tile."""
        return self._take(stage_info, cutlass.Boolean(False))

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=sage_scale_arr)
    @cute.jit
    def take_route_tile(
        self, stage_info: StageInfo, *, route_flags: Int32
    ) -> cutlass.Array:
        """Return the routed array of a block-sparse route."""
        return self._take(stage_info, self._route_is_proxy(route_flags))

    @cute.jit
    def _route_is_proxy(self, route_flags: Int32) -> cutlass.Boolean:
        """Return the route kind, resolved only where it selects a geometry."""
        route_is_proxy = cutlass.Boolean(False)
        if cutlass.const_expr(self._selects_by_route):
            route_is_proxy = cute.arch.make_warp_uniform(_route_is_proxy(route_flags))
        return route_is_proxy

    @cute.jit
    def _fill_ring(
        self, stage_info: StageInfo, route_is_proxy: cutlass.Boolean
    ) -> None:
        """Store the tile's words into the acquired slot (SMEM form).

        Each thread of the instance stores at most two words (one per round
        of the instance's threads); the pipeline commit that follows publishes
        the slot. In a mixed-geometry plan only the resource of the route's
        kind fills; the branch is CTA-uniform.
        """
        assert self.in_smem
        if self._serves(route_is_proxy):
            self._store_tile_words(stage_info)

    @cute.jit
    def _take(
        self, stage_info: StageInfo, route_is_proxy: cutlass.Boolean
    ) -> cutlass.Array:
        """Return the routed array: the lane's words in the register form, else ones.

        The entries are ones before the gather so a mixed plan's inactive
        geometry and the SMEM form leave defined padding.
        """
        words = cutlass.Array(
            Float32, self.routed_words, space=cutlass.AddressSpace.rmem
        )
        for entry in cutlass.range_constexpr(self.routed_words):
            words[entry] = Float32(1.0)
        if cutlass.const_expr(not self.in_smem):
            if self._serves(route_is_proxy):
                self._gather_lane_words(stage_info, words)
        return words

    @property
    def _selects_by_route(self) -> bool:
        """Whether the route kind selects between two geometries of this plan."""
        return self.cfg.sage_mixed_k_geometry and self.route_metadata is not None

    @cute.jit
    def _serves(self, route_is_proxy: cutlass.Boolean) -> cutlass.Boolean:
        """Whether this resource holds the route's geometry.

        A plan with one geometry is served by its one resource on every route,
        so the test folds away at compile time there.
        """
        if cutlass.const_expr(not self._selects_by_route):
            return True
        if cutlass.const_expr(self.summary):
            return route_is_proxy
        return not route_is_proxy

    # -- storage fills -----------------------------------------------------

    @cute.jit
    def _gather_lane_words(self, stage_info: StageInfo, words: cutlass.Array) -> None:
        """Take the lane's spatial half of the tile's words into ``words``.

        Word ``half * arr_size + entry`` of the ``sage_k_scale_words`` layout
        is entry ``entry`` of the lane's array, so every entry index is a
        constant and neither pass indexes registers at run time.
        """
        word_value = self._word_source(stage_info)
        arr_size = self.arr_size
        warp_grp_thread_idx = Int32(
            _decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        )
        half = _keeps_spatial_half(self.cfg, warp_grp_thread_idx)
        half_base = half * Int32(arr_size)
        for entry in cutlass.range_constexpr(arr_size):
            words[entry] = word_value(half_base + Int32(entry), half, entry)

    @cute.jit
    def _store_tile_words(self, stage_info: StageInfo) -> None:
        """Store the tile's words into the slot of the acquired producer stage."""
        word_value = self._word_source(stage_info)
        num_words = self.tile_words
        arr_size = self.arr_size
        threads = 32 * self.cfg.softmax_num_warps(self.inst_id)
        warp_grp_thread_idx = Int32(
            _decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        )
        buffer = self._ring(stage_info.context)
        slot_base = Int32(stage_info.stage_idx) * Int32(num_words)
        for round_idx in cutlass.range_constexpr((num_words + threads - 1) // threads):
            word_idx = warp_grp_thread_idx + Int32(round_idx * threads)
            if word_idx < Int32(num_words):
                buffer[slot_base + word_idx] = word_value(
                    word_idx, word_idx // Int32(arr_size), word_idx % Int32(arr_size)
                )

    @cute.jit
    def _ring(self, context: ResourceContext) -> cutlass.Array:
        """Return the ring as an FP32 SMEM array."""
        return cutlass.Array(
            context.smem_base.data_ptr() + self._alloc.offset,
            dtype=Float32,
            shape=(SAGE_K_SCALES_RING_STAGES * self.tile_words,),
            addrspace=3,
        )

    # -- word sources ------------------------------------------------------

    @cute.jit
    def _word_source(self, stage_info: StageInfo) -> WordSource:
        """Return ``word_value(word_idx, half, lane_entry)`` for the current tile.

        ``word_idx = half * arr_size + lane_entry`` names a word of the
        ``sage_k_scale_words`` layout; the source reads it from wherever the
        plan keeps it.
        """
        cfg = self.cfg
        if cutlass.const_expr(self.route_metadata is None):
            # A dense tile gathers ``k_scale`` over its token range; the load
            # clamps tokens past the sequence end.
            (
                seq_len_kv,
                _q_group_idx,
                _element_mask_end_idx,
                tile_offset_k,
                _window_start_idx,
                _is_valid_effective_tile,
                _is_masked_final_wave,
                _tile_is_unmasked,
                _rows_are_active,
            ) = resolve_keeps_tile_context(
                cfg,
                stage_info,
                inst_id=self.inst_id,
                seqlens_kv=self.seqlens_kv,
                max_seq_len_kv=self.max_seq_len_kv,
                seq_len_q=self.seq_len_q,
                q_group_idx=self.q_group_idx,
            )
            kv_head_idx, batch_idx = _logical_head_batch(
                stage_info, self.h_k_idx, self.b_idx
            )
            k_scale_addr = self.scale_tensors.k_scale_ptr.toint()

            def dense_word(word_idx: Int32, half: Int32, lane_entry) -> Float32:
                _ = word_idx
                return load_k_scale(
                    cfg,
                    k_scale_addr,
                    self.scale_tensors.k_scale_head_stride,
                    kv_head_idx=kv_head_idx,
                    batch_idx=batch_idx,
                    seq_len_kv=seq_len_kv,
                    kv_token_idx=dense_k_scale_token(
                        cfg, half, lane_entry, tile_offset_k
                    ),
                )

            return dense_word

        metadata = self.route_metadata
        stage_base = metadata._consumer_stage_base()
        if cutlass.const_expr(not self.summary):
            # The load warp staged the route's words in the
            # ``sage_k_scale_words`` layout behind the route record.
            assert metadata.staging_layout.sage_scale_words_word_offset is not None
            source = stage_base + Int32(
                metadata.staging_layout.sage_scale_words_word_offset
            )
            staged = metadata._smem_scales

            def staged_word(word_idx: Int32, half: Int32, lane_entry) -> Float32:
                _ = half, lane_entry
                return Float32(staged[source + word_idx])

            return staged_word

        # A proxy route of a mixed-geometry plan stages no scale words: the
        # summary scale of each word is gathered from the route's staged atom
        # origins, one summary per KV block.
        groups = self.groups
        num_origins = metadata.staging_layout.num_origin_words
        kv_head_idx, batch_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        scale_addr = self.scale_tensors.k_summary_scale_ptr.toint()
        scale_head_stride = self.scale_tensors.k_summary_scale_head_stride
        summary_seq_len = Int32(cfg.num_proxy_summaries)
        log2_k_block_size = Int32(log2_block_size(cfg.sage_k_summary_block_size))
        origins = metadata._smem_words

        def summary_word(word_idx: Int32, half: Int32, lane_entry) -> Float32:
            _ = word_idx
            atom_idx, token_offset = sage_word_position(cfg, half, lane_entry, groups)
            atom_idx = cute.math.min(atom_idx, Int32(num_origins - 1))
            origin = Int32(origins[stage_base + atom_idx])
            return load_k_scale(
                cfg,
                scale_addr,
                scale_head_stride,
                kv_head_idx=kv_head_idx,
                batch_idx=batch_idx,
                seq_len_kv=summary_seq_len,
                kv_token_idx=cute.math.max(origin, Int32(0)) + token_offset,
                log2_k_block_size=log2_k_block_size,
            )

        return summary_word

    # -- pass-side reads ---------------------------------------------------

    @cute.jit
    def open(
        self,
        stage_info: StageInfo,
        scale_arr: cutlass.Array,
        factor: Float32 | None,
    ):
        """Return a pass's view of the tile's words with ``factor`` applied.

        Register form: a fresh rotating copy of the routed array, scaled once
        per tile. SMEM form: the lane's half pointer into the slot the latest
        wait selected and the factor, applied per fragment on the way out.
        """
        if cutlass.const_expr(self.in_smem):
            warp_grp_thread_idx = Int32(
                _decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            )
            half_base = Int32(self.consumer_work_stage) * Int32(
                self.tile_words
            ) + _keeps_spatial_half(self.cfg, warp_grp_thread_idx) * Int32(
                self.arr_size
            )
            half_ptr = self._ring(stage_info.context).data_ptr() + half_base
            return half_ptr, factor
        arr_size = self.arr_size
        words = cutlass.Array(Float32, arr_size, space=cutlass.AddressSpace.rmem)
        for entry in cutlass.range_constexpr(arr_size):
            words[entry] = Float32(scale_arr[entry])
        if cutlass.const_expr(factor is not None):
            scale_pairs_in_place(words, factor, arr_size)
        return words

    @cute.jit
    def fragment(self, view, fragment_idx: Int32) -> cutlass.Array:
        """Return ``factor * sfK`` of one fragment's ``groups`` words.

        The SMEM form reads them from the aligned ring in 16-byte loads;
        callers issue it ahead of the fragment's score wait so the latency
        hides behind it. The register form reads the leading entries of the
        rotating array.
        """
        groups = self.groups
        values = cutlass.Array(Float32, groups, space=cutlass.AddressSpace.rmem)
        if cutlass.const_expr(self.in_smem):
            half_ptr, factor = view
            assert groups % 4 == 0
            words_ptr = half_ptr + fragment_idx * Int32(groups)
            for chunk in cutlass.range_constexpr(0, groups, 4):
                loaded = (words_ptr + Int32(chunk)).load(count=4, alignment=16)
                for elem in cutlass.range_constexpr(4):
                    values[chunk + elem] = Float32(loaded[elem])
            if cutlass.const_expr(factor is not None):
                scale_pairs_in_place(values, factor, groups)
        else:
            for group_idx in cutlass.range_constexpr(groups):
                values[group_idx] = Float32(view[group_idx])
        return values

    @cute.jit
    def advance(self, view) -> None:
        """Rotate the register array down by one fragment; the SMEM form reads in place."""
        if cutlass.const_expr(not self.in_smem):
            groups = self.groups
            for entry in cutlass.range_constexpr(self.arr_size - groups):
                view[entry] = Float32(view[entry + groups])
