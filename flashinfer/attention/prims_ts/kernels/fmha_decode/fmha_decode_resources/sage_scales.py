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

Where a tile's ``sfK`` words live during the two softmax passes is a
compile-time strategy chosen by the K block size: :class:`RegisterSageKScales`
keeps the lane's multipliers in a rotating register array,
:class:`SmemSageKScales` keeps the tile's words in a two-tile SMEM ring of the
softmax instance. Both take a tile's words through ``load_tile`` and hand a
pass one fragment's ``factor * sfK`` at a time through the same interface
(``open``, ``fragment``, ``advance``); the passes apply ``sfQ`` once per tile
themselves.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import ResourceContext, StageInfo

from ....sage import flat_scale_slot, log2_block_size
from ..fmha_decode_config import FmhaDecodeConfig
from .helpers_common import (
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _decode_gen_task_cache,
    _keeps_route_atom,
    _keeps_spatial_half,
    fmul2,
)

Constexpr = cutlass.Constexpr


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


def _groups_or_default(cfg: FmhaDecodeConfig, groups: int | None) -> int:
    return cfg.sage_k_groups_per_fragment if groups is None else groups


def sage_scale_arr_size(cfg: FmhaDecodeConfig, groups: int | None = None) -> int:
    """Return the number of ``sfK`` words one softmax lane holds per tile.

    ``groups`` is the scale groups per fragment of the tile's route kind; the
    exact-route geometry is the default.
    """
    return cfg.num_softmax_score_fragments * _groups_or_default(cfg, groups)


def sage_k_scale_words(cfg: FmhaDecodeConfig, groups: int | None = None) -> int:
    """Return the number of ``sfK`` words that cover one KV tile for every spatial half.

    Zero without Sage attention, so the block-sparse staging layout can take
    the count unconditionally. Word ``half * arr_size + f * groups + g`` holds
    the scale of group ``g`` of fragment ``f`` of the lanes in that half
    (``sage_scale_arr_size`` words per half); this is the layout of the words
    a block-sparse route stages and of the tile buffer of
    :class:`SmemSageKScales`, so a softmax thread reads its half's values with
    contiguous vector loads. ``groups`` selects the route kind's geometry.
    """
    if not cfg.use_sage_attention:
        return 0
    return cfg.keeps_spatial_halves * sage_scale_arr_size(cfg, groups)


def sage_staged_k_scale_words(cfg: FmhaDecodeConfig) -> int:
    """Return the ``sfK`` words a block-sparse route stages: the larger geometry."""
    return max(
        sage_k_scale_words(cfg, cfg.sage_k_groups_per_fragment),
        sage_k_scale_words(cfg, cfg.sage_summary_k_groups_per_fragment),
    )


@cute.jit
def sage_word_position(
    cfg: Constexpr[FmhaDecodeConfig],
    half: Int32,
    lane_entry,
    groups: Constexpr[int | None] = None,
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
    groups per fragment, the exact-route geometry by default.
    """
    groups = _groups_or_default(cfg, groups)
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
def route_scale_word_position(
    cfg: Constexpr[FmhaDecodeConfig],
    word_idx: Int32,
    groups: Constexpr[int | None] = None,
) -> tuple[Int32, Int32]:
    """Return ``sage_word_position`` of word ``word_idx`` of the staged layout."""
    arr_size = sage_scale_arr_size(cfg, groups)
    return sage_word_position(
        cfg, word_idx // Int32(arr_size), word_idx % Int32(arr_size), groups
    )


@cute.jit
def _load_f32_chunks(ptr, count: Constexpr[int]) -> cutlass.Array:
    """Return ``count`` FP32 values from an aligned SMEM pointer in 16-byte loads."""
    assert count % 4 == 0
    values = cutlass.Array(Float32, count, space=cutlass.AddressSpace.rmem)
    for chunk in cutlass.range_constexpr(0, count, 4):
        loaded = (ptr + Int32(chunk)).load(count=4, alignment=16)
        for elem in cutlass.range_constexpr(4):
            values[chunk + elem] = Float32(loaded[elem])
    return values


@cute.jit
def dense_k_scale_token(
    cfg: Constexpr[FmhaDecodeConfig], half: Int32, lane_entry, tile_offset_k: Int32
) -> Int32:
    """Return the first KV token covered by one ``sfK`` word of a dense tile.

    A dense Keeps tile is its layout atoms in order (``_keeps_score_col``), so
    the word's position (``sage_word_position``) in a route of layout atoms
    names the token.
    """
    atom_idx, token_offset = sage_word_position(cfg, half, lane_entry)
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


@dataclass
class RegisterSageKScales:
    """The lane's ``sfK`` multipliers as a rotating register array.

    K blocks of 16 tokens and larger have at most two scale groups per K32
    fragment, so a lane holds ``sage_scale_arr_size`` words per tile (eight
    for the 16-token block on KV256). The routed ``sage_scale_arr`` carries
    the raw ``sfK`` of the lane's fragment groups in fragment order. A pass
    opens it with its factor, which is applied once per tile, reads the
    leading ``groups`` entries for the fragment at hand and rotates the
    array down by one fragment, so neither the unrolled nor the rolled
    fragment loop indexes registers at run time.

    ``groups`` is the scale groups per fragment of the route kind the
    strategy serves. ``routed_words`` may exceed the lane's word count when
    the other route kind's strategy routes more entries; the extra entries
    are padding.
    """

    cfg: FmhaDecodeConfig
    groups: int
    routed_words: int = 0

    def __post_init__(self) -> None:
        self.routed_words = max(self.routed_words, self.arr_size)

    @property
    def arr_size(self) -> int:
        """Return the number of ``sfK`` words one lane holds per tile."""
        return sage_scale_arr_size(self.cfg, self.groups)

    def smem_requirements(self, owner_name: str) -> list[SmemAllocation]:
        """Return the SMEM this strategy needs: none."""
        _ = owner_name
        return []

    @cute.jit
    def load_tile(
        self,
        stage_info: StageInfo,
        word_value: Constexpr[Callable[..., Float32]],
        words: cutlass.Array,
    ) -> None:
        """Fill ``words`` (the routed array) with the lane's ``sfK`` words of the tile.

        ``word_value(word_idx, half, lane_entry)`` returns word ``word_idx =
        half * arr_size + lane_entry`` of the ``sage_k_scale_words`` layout
        from whichever coordinates it needs; the lane takes the
        ``sage_scale_arr_size`` words of its spatial half, so its entries are
        constants. Padding entries past the lane's words are set to one.
        """
        cfg = self.cfg
        arr_size = self.arr_size
        warp_grp_thread_idx = Int32(
            _decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        )
        half = _keeps_spatial_half(cfg, warp_grp_thread_idx)
        half_base = half * Int32(arr_size)
        for entry in cutlass.range_constexpr(arr_size):
            words[entry] = word_value(half_base + Int32(entry), half, entry)
        for entry in cutlass.range_constexpr(arr_size, self.routed_words):
            words[entry] = Float32(1.0)

    @cute.jit
    def open(
        self,
        stage_info: StageInfo,
        scale_arr: cutlass.Array,
        factor: Float32 | None,
    ) -> cutlass.Array:
        """Return the tile's ``factor * sfK`` words as a fresh rotating array."""
        _ = stage_info
        arr_size = self.arr_size
        words = cutlass.Array(Float32, arr_size, space=cutlass.AddressSpace.rmem)
        for entry in cutlass.range_constexpr(arr_size):
            words[entry] = Float32(scale_arr[entry])
        if cutlass.const_expr(factor is not None):
            scale_pairs_in_place(words, factor, arr_size)
        return words

    @cute.jit
    def fragment(self, words: cutlass.Array, fragment_idx: Int32) -> cutlass.Array:
        """Return the leading fragment's ``groups`` words, the factor applied."""
        _ = fragment_idx
        groups = self.groups
        values = cutlass.Array(Float32, groups, space=cutlass.AddressSpace.rmem)
        for group_idx in cutlass.range_constexpr(groups):
            values[group_idx] = Float32(words[group_idx])
        return values

    @cute.jit
    def advance(self, words: cutlass.Array) -> None:
        """Rotate the array down by one fragment."""
        groups = self.groups
        for entry in cutlass.range_constexpr(self.arr_size - groups):
            words[entry] = Float32(words[entry + groups])


@dataclass
class SmemSageKScales:
    """A softmax instance's ``sfK`` words in a two-tile SMEM ring.

    K blocks of 4 and 1 token have eight or 32 scale groups per K32
    fragment, more multipliers than a lane can hold, so a tile's words live
    in SMEM in the layout of ``sage_k_scale_words`` and a pass reads one
    fragment's groups with 16-byte broadcast loads (all lanes of a spatial
    half read the same words), scaled by the pass's factor on the way. The
    instance's warps fill a tile's words (``load_tile``, from ``k_scale`` on
    a dense tile or from the staged route words of a block-sparse route) and
    one named barrier of the instance publishes them. Consecutive tiles
    alternate between the two tile slots of the ring, so a fill never
    overwrites words the previous tile's P pass may still be reading in
    another warp. The routed ``sage_scale_arr`` holds no words here; its
    entries are padding for the other route kind's register strategy.

    ``groups`` is the scale groups per fragment of the route kind the
    strategy serves; ``name`` distinguishes the ring of a proxy-summary
    strategy from the exact-route one when a plan holds both.
    """

    cfg: FmhaDecodeConfig
    inst_id: int
    sync_barrier_id: int
    groups: int
    name: str = "sageKScales"
    routed_words: int = 1
    _alloc: SmemAllocation | None = None

    @property
    def tile_words(self) -> int:
        """Return the ``sfK`` words of one tile slot."""
        return sage_k_scale_words(self.cfg, self.groups)

    def smem_requirements(self, owner_name: str) -> list[SmemAllocation]:
        """Return the ring's allocation, registered by the owning S resource."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{owner_name}_{self.name}",
                size_bytes=2 * self.tile_words * 4,
                alignment=16,
            )
        return [self._alloc]

    @cute.jit
    def words(self, context: ResourceContext) -> cutlass.Array:
        """Return the two-tile ring as an FP32 SMEM array."""
        return cutlass.Array(
            context.smem_base.data_ptr() + self._alloc.offset,
            dtype=Float32,
            shape=(2 * self.tile_words,),
            addrspace=3,
        )

    @cute.jit
    def tile_base(self, stage_info: StageInfo) -> Int32:
        """Return the word base of the current tile's slot of the ring."""
        return (stage_info.loop_offset & Int32(1)) * Int32(self.tile_words)

    @cute.jit
    def load_tile(
        self,
        stage_info: StageInfo,
        word_value: Constexpr[Callable[..., Float32]],
        words: cutlass.Array,
    ) -> None:
        """Fill the current tile's slot, publish it, and pad the routed array.

        ``word_value(word_idx, half, lane_entry)`` returns word ``word_idx =
        half * arr_size + lane_entry`` of the ``sage_k_scale_words`` layout
        from whichever coordinates it needs. Each lane of the instance stores
        at most two words (one per round of the instance's threads); the named
        barrier publishes the slot to both passes.
        """
        cfg = self.cfg
        num_words = self.tile_words
        arr_size = sage_scale_arr_size(cfg, self.groups)
        threads = 32 * cfg.softmax_num_warps(self.inst_id)
        warp_grp_thread_idx = Int32(
            _decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        )
        buffer = self.words(stage_info.context)
        tile_base = self.tile_base(stage_info)
        for round_idx in cutlass.range_constexpr((num_words + threads - 1) // threads):
            word_idx = warp_grp_thread_idx + Int32(round_idx * threads)
            if word_idx < Int32(num_words):
                buffer[tile_base + word_idx] = word_value(
                    word_idx, word_idx // Int32(arr_size), word_idx % Int32(arr_size)
                )
        prims.barrier_cta_sync(self.sync_barrier_id, thread_count=threads)
        for entry in cutlass.range_constexpr(self.routed_words):
            words[entry] = Float32(1.0)

    @cute.jit
    def open(
        self,
        stage_info: StageInfo,
        scale_arr: cutlass.Array,
        factor: Float32 | None,
    ) -> tuple[cutlass.Array, Int32, Float32 | None]:
        """Return the pass's view: the ring, the lane's half base and the factor."""
        _ = scale_arr
        warp_grp_thread_idx = Int32(
            _decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        )
        half_base = self.tile_base(stage_info) + _keeps_spatial_half(
            self.cfg, warp_grp_thread_idx
        ) * Int32(sage_scale_arr_size(self.cfg, self.groups))
        return self.words(stage_info.context), half_base, factor

    @cute.jit
    def fragment(self, view: tuple, fragment_idx: Int32) -> cutlass.Array:
        """Return ``factor * sfK`` of one fragment's groups.

        Callers issue it ahead of the fragment's score wait so the SMEM
        latency hides behind it.
        """
        buffer, half_base, factor = view
        groups = self.groups
        values = _load_f32_chunks(
            buffer.data_ptr() + half_base + fragment_idx * Int32(groups), groups
        )
        if cutlass.const_expr(factor is not None):
            scale_pairs_in_place(values, factor, groups)
        return values

    @cute.jit
    def advance(self, view: tuple) -> None:
        """Nothing rotates: every fragment reads its own words."""
        _ = view


SageKScales = RegisterSageKScales | SmemSageKScales


def make_sage_k_scales(
    cfg: FmhaDecodeConfig, *, inst_id: int, sync_barrier_id: int
) -> tuple[SageKScales | None, SageKScales | None]:
    """Return one softmax instance's ``sfK`` strategies for exact and proxy routes.

    Both entries are ``None`` without Sage attention and the same object
    unless the plan's summary scales use another K block size than its token
    scales (``sage_mixed_k_geometry``); then the proxy strategy is built for
    the summary geometry, with its own SMEM ring and named barrier
    (``sync_barrier_id + 2``) if it lives in SMEM, and both route the larger
    ``sage_scale_arr``.
    """
    if not cfg.use_sage_attention:
        return None, None

    def build(groups: int, barrier_id: int, name: str) -> SageKScales:
        if cfg.sage_k_scales_in_smem_for(groups):
            return SmemSageKScales(cfg, inst_id, barrier_id, groups, name=name)
        return RegisterSageKScales(cfg, groups)

    exact = build(cfg.sage_k_groups_per_fragment, sync_barrier_id, "sageKScales")
    if not cfg.sage_mixed_k_geometry:
        return exact, exact
    proxy = build(
        cfg.sage_summary_k_groups_per_fragment,
        sync_barrier_id + 2,
        "sageSummaryKScales",
    )
    routed_words = max(exact.routed_words, proxy.routed_words)
    exact.routed_words = routed_words
    proxy.routed_words = routed_words
    return exact, proxy


@cute.jit
def block_sparse_k_scale_source(
    cfg: Constexpr[FmhaDecodeConfig],
    *,
    k_scale_ptr: cute.Pointer,
    k_scale_head_stride: Int32,
    k_summary_scale_ptr: cute.Pointer | None,
    k_summary_scale_head_stride: Int32 | None,
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
    scale_addr = k_scale_ptr.toint()
    head_stride = k_scale_head_stride
    seq_len = seq_len_kv
    log2_block = Int32(log2_block_size(cfg.sage_k_block_size))
    if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
        if route_is_proxy:
            scale_addr = k_summary_scale_ptr.toint()
            head_stride = k_summary_scale_head_stride
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
    v_scale_ptr: cute.Pointer,
    v_mean_ptr: cute.Pointer | None,
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
    v_scale_addr = v_scale_ptr.toint()
    head_base = kv_head_idx * Int32(cfg.headdim)
    for base in cutlass.range_constexpr(0, cfg.headdim, num_threads):
        channel = Int32(base) + thread_idx
        staged[channel] = _load_scale(v_scale_addr, head_base + channel)
        if cutlass.const_expr(cfg.sage_v_mean):
            staged[channel + Int32(cfg.headdim)] = _load_scale(
                v_mean_ptr.toint(), head_base + channel
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
