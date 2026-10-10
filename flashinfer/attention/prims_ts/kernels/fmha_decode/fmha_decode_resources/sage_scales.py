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

``sfQ`` and ``sfK`` reach the softmax from the images the Sage prepare writes
before attention (``_SageQScaleImageLayout``, ``_SageKScaleImageLayout``), in
the order the softmax reads them; the prepare resolves the flat layout
(``flat_scale_slot`` of :mod:`flashinfer.attention.prims_ts.sage`) and gives
Q rows past the valid row count and K tokens past the sequence end the last
valid scale, so masked scores keep a finite scale and exponentiate to zero.
Each softmax lane loads its row's ``sfQ`` word once per work tile
(``load_q_scale``).

:class:`SageKScalesResource` is one softmax instance's ring of ``sfK``
tiles, which the load warp produces after each K tile with ``cp.async``
copies under the ring's ``AsyncLoad`` commit: the chunks of the tile's
atoms, a dense tile's or a block-sparse route's, from the
``_SageKScaleImageLayout`` image the Sage prepare writes
(``_copy_atom_chunks``). Both softmax passes read the waited stage in place,
one K32 fragment at a time with loads of at most 16 bytes, so attention never
addresses ``k_scale`` itself.

:class:`SageVScalesResource` stages the work tile's per-channel V scales
(and means) in SMEM once per tile, while no output is pending, so the
epilogue scales column ``c`` by ``sfV[c]`` and adds ``v_mean[c]`` without
waiting on global loads.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    producer_work,
)

from flashinfer.utils import round_up

from ...._block_sparse.prepared import (
    _PREPARED_ROUTE_IS_PROXY_FLAG,
    _SAGE_IMAGE_ATOM_TOKENS,
    _SAGE_IMAGE_PIECE_WORDS,
    _BlockSparseRouteLayout,
)
from ....sage import flat_scale_slot
from ...stage import FmhaStage
from ..fmha_decode_config import FmhaDecodeConfig
from ..fmha_decode_constants import KV_KIND_K
from .helpers_common import (
    DecodeGenResourceBase,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _decode_gen_task_cache,
    _keeps_spatial_half,
    _logical_head_batch,
    _warp_broadcast_i32,
    fmul2,
)
from .helpers_kv_tile_idx import (
    _local_kv_tile_idx_for_section,
    resolve_keeps_tile_context,
)

Constexpr = cutlass.Constexpr


@dataclass(eq=False)
class SageScaleTensors:
    """The plan's Sage scale tensors, shared by every resource that reads one.

    ``sfQ`` and ``sfK`` are the plan's prepared images
    (``_SageQScaleImageLayout`` at ``q_scale_image_ptr``,
    ``_SageKScaleImageLayout`` at ``k_scale_image_ptr``); the V scales and
    means are ``[Hkv, D]`` FP32, ``v_mean_ptr`` being ``None`` without a
    channel mean. The dataclass is not frozen because the DSL replaces frozen
    dataclasses with proxies inside traced dynamic branches, which changes
    the traced structure of the holding resource.
    """

    q_scale_image_ptr: cute.Pointer
    k_scale_image_ptr: cute.Pointer
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
def load_flat_scale(
    scale_addr: Int64,
    head_stride: Int32,
    *,
    head_idx: Int32,
    batch_idx: Int32,
    seq_len: Int32,
    token_idx: Int32,
    log2_block: Int32,
) -> Float32:
    """Return one head's scale of one token from a flat-layout scale array.

    A token past ``seq_len`` takes the last valid slot.
    """
    token_idx = cute.math.min(token_idx, seq_len - Int32(1))
    slot = flat_scale_slot(batch_idx, token_idx, seq_len, log2_block)
    return _load_scale(scale_addr, head_idx * head_stride + slot)


@cute.jit
def load_q_scale(
    cfg: Constexpr[FmhaDecodeConfig],
    scales: SageScaleTensors,
    *,
    sequence_idx: Int32,
    q_group_idx: Int32,
    row_idx: Int32,
) -> Float32:
    """Return ``sfQ`` of row ``row_idx`` of Q tile ``q_group_idx`` from the prepared image.

    ``sequence_idx = b * Hkv + h`` selects the sequence and KV head. The word
    already holds the scale of the row's token and Q head, the last valid
    row's for a row past the tile's valid rows (``_SageQScaleImageLayout``).
    """
    image = cfg.sage_q_scale_image
    word = (
        sequence_idx * Int32(image.sequence_words)
        + q_group_idx * Int32(image.tile_size_q)
        + row_idx
    )
    return _load_scale(scales.q_scale_image_ptr.toint(), word)


def sage_k_chunk_words(cfg: FmhaDecodeConfig, proxy: bool = False) -> int:
    """Return one route kind's ``sfK`` words of a 64-token atom, padded to whole 16-byte pieces.

    Word ``(f % 2) * groups + g`` is group ``g`` of the atom's fragment
    ``f % 2``; only the one-group geometry pads.
    """
    words = cfg.keeps_fragments_per_atom * cfg.sage_k_groups_per_fragment(proxy)
    return round_up(words, _SAGE_IMAGE_PIECE_WORDS)


def sage_k_tile_words(cfg: FmhaDecodeConfig) -> int:
    """Return the ``sfK`` words of one KV tile: one chunk per atom of the larger kind.

    A route kind lays its chunks out contiguously: atom ``a``, held by
    spatial half ``h`` at position ``p``
    (``FmhaDecodeConfig.keeps_route_atom_owner``), owns chunk ``h *
    atoms_per_half + p``, so each half's words are contiguous and fragment
    ``f`` of an unpadded half starts at word ``f * groups`` of it.
    """
    chunk_words = max(sage_k_chunk_words(cfg), sage_k_chunk_words(cfg, proxy=True))
    return cfg.tile_size_kv // cfg.keeps_atom_tokens * chunk_words


@cute.jit
def _copy_atom_chunks(
    cfg: Constexpr[FmhaDecodeConfig],
    scales: SageScaleTensors,
    stage_words: cute.Pointer,
    kind_words: Int32,
    atom_chunk: Constexpr,
    num_atoms: Constexpr[int],
    proxy: Constexpr[bool],
) -> None:
    """Copy ``num_atoms`` atoms' chunks of one route kind into a ring stage.

    Lane ``l`` copies 16-byte piece ``l`` (and ``l + 32``, ...) of the chunks
    with ``cp.async``, so the TMA unit stays free for K and V. Atom ``a``
    reads image chunk ``atom_chunk(a)`` of the kind's chunk size, counted
    from image word ``kind_words``, into the stage slot of its spatial half
    and position (``sage_k_tile_words``).
    """
    chunk_words = sage_k_chunk_words(cfg, proxy)
    pieces_per_atom = chunk_words // _SAGE_IMAGE_PIECE_WORDS
    num_pieces = num_atoms * pieces_per_atom
    lane_idx = cute.arch.thread_idx()[0] & Int32(31)
    for round_idx in cutlass.range_constexpr((num_pieces + 31) // 32):
        piece = lane_idx + Int32(round_idx * 32)
        atom = piece // Int32(pieces_per_atom)
        # Every lane resolves its piece's chunk: a route's lookup may shuffle.
        chunk_idx = atom_chunk(atom)
        half, position = cfg.keeps_route_atom_owner(atom)
        slot = half * Int32(num_atoms // cfg.keeps_spatial_halves) + position
        word_in_chunk = (piece % Int32(pieces_per_atom)) * Int32(
            _SAGE_IMAGE_PIECE_WORDS
        )
        if piece < Int32(num_pieces):
            prims.cp_async_shared_global(
                stage_words + slot * Int32(chunk_words) + word_in_chunk,
                scales.k_scale_image_ptr
                + kind_words
                + chunk_idx * Int32(chunk_words)
                + word_in_chunk,
                _SAGE_IMAGE_PIECE_WORDS * 4,
                "cg",
            )


@cute.jit
def _route_atom_chunk(
    route_layout: Constexpr[_BlockSparseRouteLayout],
    pieces_per_atom: Constexpr[int],
    first_chunk: Int32,
    resolved_record_word: Int32,
    resolved_origin1: Int32,
    atom: Int32,
) -> Int32:
    """Return the image chunk of route atom ``atom``: ``first_chunk + origin / 64``.

    ``resolved_*`` are the route's record words as
    ``SmemBlockSparseKvMetadataResource.resolve_route`` returns them. An
    invalid atom (origin ``-1``) takes the kind's first chunk, whose finite
    scales its masked scores never use.
    """
    num_atoms = route_layout.logical_origins_per_route
    if cutlass.const_expr(num_atoms == 2 and not route_layout.uses_one_warp_transport):
        # Two-origin records travel as two warp-uniform scalars.
        origin = Int32(resolved_record_word)
        if atom != Int32(0):
            origin = Int32(resolved_origin1)
    elif cutlass.const_expr(pieces_per_atom == 1):
        # Lane ``a`` copies atom ``a`` and holds its record word.
        origin = Int32(resolved_record_word)
    else:
        origin = cute.arch.shuffle_sync(resolved_record_word, atom)
    return first_chunk + cute.math.max(origin, Int32(0)) // Int32(
        _SAGE_IMAGE_ATOM_TOKENS
    )


@cute.jit
def copy_route_k_scale_chunks(
    cfg: Constexpr[FmhaDecodeConfig],
    route_layout: Constexpr[_BlockSparseRouteLayout],
    scales: SageScaleTensors,
    *,
    stage_words: cute.Pointer,
    sequence_idx: Int32,
    resolved_record_word: Int32,
    resolved_origin1: Int32,
) -> None:
    """Copy a block-sparse route's ``sfK`` chunks from the prepared image.

    ``stage_words`` points at a ring stage in the ``sage_k_tile_words``
    layout. Each route atom reads the chunk of its origin from sequence
    ``sequence_idx = b * Hkv + h`` of the image, in its route kind's
    geometry (``_route_atom_chunk``).
    """
    image = cfg.sage_k_scale_image(cfg.static_seq_len_kv)
    assert image.exact.chunk_words == sage_k_chunk_words(cfg)
    assert image.summary is None or image.summary.chunk_words == sage_k_chunk_words(
        cfg, proxy=True
    )
    num_atoms = route_layout.logical_origins_per_route
    chunk_words = sage_k_chunk_words(cfg)
    pieces_per_atom = chunk_words // _SAGE_IMAGE_PIECE_WORDS
    is_proxy_route = cutlass.Boolean(False)
    if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
        route_flags = _warp_broadcast_i32(
            resolved_record_word, route_layout.route_flags_word_offset
        )
        is_proxy_route = (route_flags & Int32(_PREPARED_ROUTE_IS_PROXY_FLAG)) != Int32(
            0
        )
    if cutlass.const_expr(cfg.sage_mixed_k_geometry):
        # The kinds' chunk sizes differ, so the copy counts from the kind's
        # first image word.
        sequence_words = sequence_idx * Int32(image.sequence_words)
        if is_proxy_route:
            summary_pieces = image.summary.chunk_words // _SAGE_IMAGE_PIECE_WORDS
            _copy_atom_chunks(
                cfg,
                scales,
                stage_words,
                sequence_words + Int32(image.summary.word_offset),
                lambda atom: _route_atom_chunk(
                    route_layout,
                    summary_pieces,
                    Int32(0),
                    resolved_record_word,
                    resolved_origin1,
                    atom,
                ),
                num_atoms=num_atoms,
                proxy=True,
            )
        else:
            _copy_atom_chunks(
                cfg,
                scales,
                stage_words,
                sequence_words,
                lambda atom: _route_atom_chunk(
                    route_layout,
                    pieces_per_atom,
                    Int32(0),
                    resolved_record_word,
                    resolved_origin1,
                    atom,
                ),
                num_atoms=num_atoms,
                proxy=False,
            )
    else:
        first_chunk = sequence_idx * Int32(image.sequence_words // chunk_words)
        if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
            # A proxy route reads the summaries' chunks behind the K tokens'.
            if is_proxy_route:
                first_chunk = first_chunk + Int32(image.exact.atoms)
        _copy_atom_chunks(
            cfg,
            scales,
            stage_words,
            Int32(0),
            lambda atom: _route_atom_chunk(
                route_layout,
                pieces_per_atom,
                first_chunk,
                resolved_record_word,
                resolved_origin1,
                atom,
            ),
            num_atoms=num_atoms,
            proxy=False,
        )


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

    Each of the ``num_threads`` callers moves ``headdim / num_threads``
    channels. Callers order these writes before the epilogue reads, and the
    next tile's writes after the last read.
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


@dataclass(kw_only=True)
class SageKScalesResource(DecodeGenResourceBase):
    """One softmax instance's ring of ``sfK`` tiles, produced by the load warp.

    Each stage holds one KV tile's words in the ``sage_k_tile_words`` layout.
    The load warp fills it after the tile's K copy with ``cp.async`` copies of
    the tile's atom chunks from the prepared image, a block-sparse route's
    (``copy_route``) or a dense tile's (``copy_tile``), and the pipeline's
    ``AsyncLoad`` commit completes the stage once they land. The softmax reads
    the waited stage in place and releases it after its last pass that reads
    the words.
    """

    cfg: Constexpr[FmhaDecodeConfig] = None
    inst_id: Constexpr[int] = 0
    route_layout: Constexpr[_BlockSparseRouteLayout | None] = None
    seqlens_kv: cute.Pointer | None = None
    max_seq_len_kv: Int32 = None
    seq_len_q: Int32 = None
    q_group_idx: Int32 | None = None
    h_k_idx: Int32 | None = None
    b_idx: Int32 | None = None
    num_heads_kv: Int32 | None = None
    scale_tensors: SageScaleTensors | None = None
    _alloc: Constexpr[SmemAllocation | None] = None

    def __post_init__(self) -> None:
        assert self.pipeline_config is not None
        assert (self.route_layout is not None) == self.cfg.use_block_sparse
        super().__post_init__()

    @property
    def tile_words(self) -> int:
        """Return the ``sfK`` words of one tile in a ring stage."""
        return sage_k_tile_words(self.cfg)

    def half_words(self, proxy: bool = False) -> int:
        """Return one route kind's stage words of one spatial half's atoms."""
        atoms = self.cfg.tile_size_kv // self.cfg.keeps_atom_tokens
        return (
            atoms // self.cfg.keeps_spatial_halves * sage_k_chunk_words(self.cfg, proxy)
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the ring."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=self.name,
                size_bytes=self.pipeline_config.num_stages * self.tile_words * 4,
                alignment=16,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """The words live in SMEM only."""
        return []

    # -- producer steps ----------------------------------------------------

    @cute.jit
    def _stage_words(self, stage_info: StageInfo) -> cute.Pointer:
        """Return the acquired stage's first word as an FP32 SMEM pointer."""
        return self._ring(stage_info.context).data_ptr() + Int32(
            stage_info.stage_idx
        ) * Int32(self.tile_words)

    @producer_work
    @cute.jit
    def copy_route(
        self,
        stage_info: StageInfo,
        *,
        resolved_record_word: Int32,
        resolved_origin1: Int32,
    ) -> None:
        """Copy a block-sparse route's words into the acquired ring stage.

        The load warp issues the ``cp.async`` copies only; the pipeline's
        ``AsyncLoad`` commit completes the stage once they land.
        """
        assert self.route_layout is not None
        kv_head_idx, batch_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        copy_route_k_scale_chunks(
            self.cfg,
            self.route_layout,
            self.scale_tensors,
            stage_words=self._stage_words(stage_info),
            sequence_idx=batch_idx * self.num_heads_kv + kv_head_idx,
            resolved_record_word=resolved_record_word,
            resolved_origin1=resolved_origin1,
        )

    @producer_work
    @cute.jit
    def copy_tile(
        self, stage_info: StageInfo, *, section: Constexpr[FmhaStage]
    ) -> None:
        """Copy a dense tile's atom chunks from the image into the acquired ring stage.

        The tile is the instance's K tile of ``section``, resolved as the
        softmax resolves the tile it reads. Atom ``a`` of the tile reads the
        chunk of its first token, ``tile_offset_k + a * 64``, clamped to the
        sequence's last chunk: a tile or atom past the sequence end has
        masked scores. The pipeline's ``AsyncLoad`` commit completes the
        stage once the copies land.
        """
        assert self.route_layout is None
        cfg = self.cfg
        (
            _seq_len_kv,
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
            local_tile_idx=_local_kv_tile_idx_for_section(
                cfg, stage_info, self.inst_id, KV_KIND_K, section
            ),
        )
        kv_head_idx, batch_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        # A dense plan's image holds the K-token kind alone.
        image = cfg.sage_k_scale_image(cfg.static_seq_len_kv)
        assert image.summary is None
        atoms = image.exact.atoms
        first_chunk = (batch_idx * self.num_heads_kv + kv_head_idx) * Int32(atoms)
        tile_atom = tile_offset_k // Int32(_SAGE_IMAGE_ATOM_TOKENS)
        _copy_atom_chunks(
            cfg,
            self.scale_tensors,
            self._stage_words(stage_info),
            Int32(0),
            lambda atom: first_chunk
            + cute.math.min(tile_atom + atom, Int32(atoms - 1)),
            num_atoms=cfg.tile_size_kv // cfg.keeps_atom_tokens,
            proxy=False,
        )

    # -- storage helpers ---------------------------------------------------

    @cute.jit
    def _lane_half(self, stage_info: StageInfo) -> Int32:
        """Return the spatial half of the calling softmax thread."""
        warp_grp_thread_idx = Int32(
            _decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        )
        return _keeps_spatial_half(self.cfg, warp_grp_thread_idx)

    @cute.jit
    def _ring(self, context: ResourceContext) -> cutlass.Array:
        """Return the ring as an FP32 SMEM array."""
        return cutlass.Array(
            context.smem_base.data_ptr() + self._alloc.offset,
            dtype=Float32,
            shape=(self.pipeline_config.num_stages * self.tile_words,),
            addrspace=3,
        )

    # -- pass-side reads ---------------------------------------------------

    @cute.jit
    def open(
        self,
        stage_info: StageInfo,
        factor: Float32 | None,
        proxy_kind: Constexpr[bool] = False,
    ):
        """Return a pass's view of the waited stage: the lane's half pointer and ``factor``."""
        half_ptr = (
            self._ring(stage_info.context).data_ptr()
            + Int32(self.consumer_work_stage) * Int32(self.tile_words)
            + self._lane_half(stage_info) * Int32(self.half_words(proxy_kind))
        )
        return half_ptr, factor

    @cute.jit
    def fragment(
        self, view, fragment_idx: Int32, proxy_kind: Constexpr[bool] = False
    ) -> cutlass.Array:
        """Return ``factor * sfK`` of one fragment's ``groups`` words.

        Fragment ``f`` sits in chunk ``f // 2`` of the half, after the groups
        of fragment ``f - 1`` when ``f`` is odd, so an unpadded chunk puts it
        at word ``f * groups``; the loads, at most 16 bytes each, issue ahead
        of the fragment's score wait in the callers.
        """
        groups = self.cfg.sage_k_groups_per_fragment(proxy_kind)
        values = cutlass.Array(Float32, groups, space=cutlass.AddressSpace.rmem)
        half_ptr, factor = view
        fragments_per_atom = self.cfg.keeps_fragments_per_atom
        chunk_words = sage_k_chunk_words(self.cfg, proxy_kind)
        if cutlass.const_expr(chunk_words == fragments_per_atom * groups):
            words_ptr = half_ptr + fragment_idx * Int32(groups)
        else:
            atom_position = fragment_idx // Int32(fragments_per_atom)
            words_ptr = (
                half_ptr
                + atom_position * Int32(chunk_words)
                + fragment_idx % Int32(fragments_per_atom) * Int32(groups)
            )
        width = min(groups, 4)
        for chunk in cutlass.range_constexpr(0, groups, width):
            loaded = (words_ptr + Int32(chunk)).load(count=width, alignment=width * 4)
            for elem in cutlass.range_constexpr(width):
                values[chunk + elem] = Float32(loaded[elem])
        if cutlass.const_expr(factor is not None):
            if cutlass.const_expr(groups == 1):
                values[0] = Float32(values[0]) * factor
            else:
                scale_pairs_in_place(values, factor, groups)
        return values


@dataclass(kw_only=True)
class SageVScalesResource(DecodeGenResourceBase):
    """The staged ``sfV`` and ``v_mean`` of the work tile's KV head.

    The pipeline is a one-stage pipeline of the correction warps.
    """

    cfg: Constexpr[FmhaDecodeConfig] = None
    scale_tensors: SageScaleTensors | None = None
    h_k_idx: Int32 | None = None
    b_idx: Int32 | None = None
    _alloc: Constexpr[SmemAllocation | None] = None

    @property
    def entries(self) -> int:
        """Return the FP32 entries of the block: the scales, then the means."""
        return staged_v_channel_scale_entries(self.cfg)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the one-stage block of scales and means."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=self.name,
                size_bytes=self.entries * 4,
                alignment=16,
            )
        return [self._alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """The scales live in SMEM only."""
        return []

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def stage_tile(self, stage_info: StageInfo) -> None:
        """Copy the work tile's KV head scales and means into the acquired block."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        kv_head_idx, _ = _logical_head_batch(stage_info, self.h_k_idx, self.b_idx)
        stage_v_channel_scales(
            cfg,
            self.scale_tensors,
            self.staged(stage_info.context),
            kv_head_idx=kv_head_idx,
            thread_idx=task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX],
            num_threads=cfg.correction_barrier_threads,
        )

    @cute.jit
    def staged(self, context: ResourceContext) -> cutlass.Array:
        """Return the block as an FP32 SMEM array."""
        return cutlass.Array(
            context.smem_base.data_ptr() + self._alloc.offset,
            dtype=Float32,
            shape=(self.entries,),
            addrspace=3,
        )

    @cute.jit
    def channel_scales(
        self, staged: cutlass.Array, *, first_col: Int32, count: Constexpr[int]
    ) -> tuple[cutlass.Array, cutlass.Array]:
        """Return ``count`` scales and means from ``first_col`` of the waited block.

        Without ``sage_v_mean`` the means are zero so callers apply one fused
        multiply-add.
        """
        return load_staged_v_channel_scales(
            self.cfg, staged, first_col=first_col, count=count
        )
