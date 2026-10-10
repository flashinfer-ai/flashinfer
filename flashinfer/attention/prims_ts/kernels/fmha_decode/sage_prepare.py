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

"""Prepare the Sage scale images of a dense or block-sparse plan.

Sage attention reads ``sfK`` from the ``_SageKScaleImageLayout`` image
alone: per sequence and KV head, one chunk per 64-token atom holding the
atom's scale groups in the order the softmax reads them, first for the K
tokens, then for a proxy plan's block summaries. It reads ``sfQ`` from the
``_SageQScaleImageLayout`` image alone: per sequence and KV head, one word per
row of each Q tile, the scale of the token and Q head the softmax maps the
row to. This kernel writes both images from the flat ``q_scale``, ``k_scale``
and ``k_summary_scale`` tensors ahead of the attention launch (and of a
block-sparse plan's route prepare); one thread writes one 16-byte piece.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_drv

from flashinfer.utils import ceil_div

from ..._block_sparse.prepared import (
    _SAGE_IMAGE_ATOM_TOKENS,
    _SAGE_IMAGE_PIECE_WORDS,
    _SageKScaleImageKind,
    _SageKScaleImageLayout,
)
from ...sage import log2_block_size
from .fmha_decode_config import FmhaDecodeConfig
from .fmha_decode_resources.helpers_common import (
    _q_row_token_and_local_head,
    _q_tile_valid_rows_for_seq,
)
from .fmha_decode_resources.sage_scales import load_flat_scale

_THREADS_PER_CTA = 128


@dataclass
class _KindSource:
    """One route kind's chunk geometry and source scales, as a piece writer reads them.

    The dataclass is not frozen because the writer selects one of the two
    kinds inside a traced branch, where the DSL rebuilds a frozen dataclass as
    a proxy.
    """

    # Pieces of one atom's chunk.
    chunk_pieces: cutlass.Int32
    # Tokens from one scale group's token to the next group's.
    group_tokens: cutlass.Int32
    # Words of a chunk that hold scales; the rest are zero padding.
    used_words: cutlass.Int32
    # log2 of the source tensor's block size.
    log2_block: cutlass.Int32
    # Base address and per-head slot count of the flat source tensor.
    scale_addr: cutlass.Int64
    head_stride: cutlass.Int32
    # Tokens of the kind: K tokens, or block summaries.
    length: cutlass.Int32

    @staticmethod
    def create(kind: _SageKScaleImageKind, scales: cute.Tensor) -> "_KindSource":
        return _KindSource(
            chunk_pieces=cutlass.Int32(kind.chunk_words // _SAGE_IMAGE_PIECE_WORDS),
            group_tokens=cutlass.Int32(kind.group_tokens),
            used_words=cutlass.Int32(kind.used_words),
            log2_block=cutlass.Int32(log2_block_size(kind.source_block_size)),
            scale_addr=scales.iterator.toint(),
            head_stride=cutlass.Int32(scales.shape[1]),
            length=cutlass.Int32(kind.length),
        )


@cute.jit
def _sequence_of_piece(
    piece_idx: cutlass.Int32,
    sequence_pieces: cutlass.Constexpr[int],
    num_kv_heads: cutlass.Constexpr[int],
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """Return the batch, KV head and sequence-local piece of image piece ``piece_idx``."""

    sequence_idx = piece_idx // cutlass.Int32(sequence_pieces)
    piece_in_sequence = piece_idx - sequence_idx * cutlass.Int32(sequence_pieces)
    batch_idx = sequence_idx // cutlass.Int32(num_kv_heads)
    head_idx = sequence_idx - batch_idx * cutlass.Int32(num_kv_heads)
    return batch_idx, head_idx, piece_in_sequence


@cute.jit
def _store_piece(
    image: cute.Tensor, piece_idx: cutlass.Int32, words: cutlass.Array
) -> None:
    """Store four words as 16-byte piece ``piece_idx`` of ``image``."""

    piece_ptr = cutlass.inttoptr(
        image.iterator.toint()
        + cutlass.Int64(piece_idx) * cutlass.Int64(_SAGE_IMAGE_PIECE_WORDS * 4),
        mem_space=1,
        dtype=cutlass.Float32,
    )
    piece_ptr.store(
        words.data_ptr().load(count=_SAGE_IMAGE_PIECE_WORDS, alignment=4),
        alignment=_SAGE_IMAGE_PIECE_WORDS * 4,
    )


@cute.jit
def _write_k_image_piece(
    k_scale: cute.Tensor,
    k_summary_scale: cute.Tensor | None,
    k_scale_image: cute.Tensor,
    piece_idx: cutlass.Int32,
    image: cutlass.Constexpr[_SageKScaleImageLayout],
    num_kv_heads: cutlass.Constexpr[int],
) -> None:
    """Write 16-byte piece ``piece_idx`` of the ``sfK`` image: four words of one atom's chunk.

    The piece lies in sequence ``b * Hkv + h`` of the image, in the chunks of
    the kind its position selects: the K tokens from ``k_scale``, or the
    block summaries from ``k_summary_scale`` past the sequence's K-token
    chunks. Word ``w`` of a chunk is the kind's scale of the atom's token ``w
    * group_tokens``; a token past the kind's length takes the last scale,
    and padding words are zero.
    """

    batch_idx, head_idx, piece_in_sequence = _sequence_of_piece(
        piece_idx, image.sequence_words // _SAGE_IMAGE_PIECE_WORDS, num_kv_heads
    )
    kind = _KindSource.create(image.exact, k_scale)
    piece_in_kind = piece_in_sequence
    if cutlass.const_expr(image.summary is not None):
        summary_first_piece = image.summary.word_offset // _SAGE_IMAGE_PIECE_WORDS
        if piece_in_sequence >= cutlass.Int32(summary_first_piece):
            kind = _KindSource.create(image.summary, k_summary_scale)
            piece_in_kind = piece_in_sequence - cutlass.Int32(summary_first_piece)
    atom_idx = piece_in_kind // kind.chunk_pieces
    first_word = (piece_in_kind - atom_idx * kind.chunk_pieces) * cutlass.Int32(
        _SAGE_IMAGE_PIECE_WORDS
    )

    words = cutlass.Array(
        cutlass.Float32, _SAGE_IMAGE_PIECE_WORDS, space=cutlass.AddressSpace.rmem
    )
    for elem in cutlass.range_constexpr(_SAGE_IMAGE_PIECE_WORDS):
        word = first_word + cutlass.Int32(elem)
        words[elem] = cutlass.Float32(0.0)
        # Only a one-group chunk pads, and it is a single piece.
        if word < kind.used_words:
            words[elem] = load_flat_scale(
                kind.scale_addr,
                kind.head_stride,
                head_idx=head_idx,
                batch_idx=batch_idx,
                seq_len=kind.length,
                token_idx=atom_idx * cutlass.Int32(_SAGE_IMAGE_ATOM_TOKENS)
                + word * kind.group_tokens,
                log2_block=kind.log2_block,
            )
    _store_piece(k_scale_image, piece_idx, words)


@cute.jit
def _write_q_image_piece(
    q_scale: cute.Tensor,
    q_scale_image: cute.Tensor,
    piece_idx: cutlass.Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    num_kv_heads: cutlass.Constexpr[int],
) -> None:
    """Write 16-byte piece ``piece_idx`` of the ``sfQ`` image: four rows of one Q tile.

    The piece lies in sequence ``b * Hkv + h`` of the image, in the tile its
    position selects. Row ``r`` of the tile takes the Q scale of the token
    and Q head the softmax maps the row to (``_q_row_token_and_local_head``);
    a row past the tile's valid rows (``_q_tile_valid_rows_for_seq``) takes
    the last valid row's.
    """

    image = cfg.sage_q_scale_image
    batch_idx, kv_head_idx, piece_in_sequence = _sequence_of_piece(
        piece_idx, image.sequence_words // _SAGE_IMAGE_PIECE_WORDS, num_kv_heads
    )
    tile_pieces = cutlass.Int32(image.tile_size_q // _SAGE_IMAGE_PIECE_WORDS)
    tile_idx = piece_in_sequence // tile_pieces
    first_row = (piece_in_sequence - tile_idx * tile_pieces) * cutlass.Int32(
        _SAGE_IMAGE_PIECE_WORDS
    )
    # The row helpers take the plan's heads per KV head as the packed row
    # count and its fixed Q length; Sage plans have no variable Q lengths.
    heads_q_per_kv = cutlass.Int32(cfg.heads_q_per_kv)
    seq_len_q = cutlass.Int32(image.seq_len_q)
    last_row = _q_tile_valid_rows_for_seq(
        cfg, heads_q_per_kv, tile_idx, seq_len_q
    ) - cutlass.Int32(1)

    words = cutlass.Array(
        cutlass.Float32, _SAGE_IMAGE_PIECE_WORDS, space=cutlass.AddressSpace.rmem
    )
    for elem in cutlass.range_constexpr(_SAGE_IMAGE_PIECE_WORDS):
        row = cute.math.min(first_row + cutlass.Int32(elem), last_row)
        q_token_idx, local_head_idx = _q_row_token_and_local_head(
            cfg, heads_q_per_kv, tile_idx, row
        )
        words[elem] = load_flat_scale(
            q_scale.iterator.toint(),
            cutlass.Int32(q_scale.shape[1]),
            head_idx=kv_head_idx * heads_q_per_kv + local_head_idx,
            batch_idx=batch_idx,
            seq_len=seq_len_q,
            token_idx=q_token_idx,
            log2_block=cutlass.Int32(log2_block_size(image.q_block_size)),
        )
    _store_piece(q_scale_image, piece_idx, words)


class _PrepareSageScaleImages:
    """Write a Sage plan's ``sfK`` and ``sfQ`` images from its flat scale tensors.

    The images cover ``batch_size * num_kv_heads`` sequences; the grid gives
    each of their 16-byte pieces one thread, the K image's pieces first and
    the Q image's after them. The kernel is an ordinary launch on the plan's
    stream. A dense plan's attention launch follows it in stream order; a
    block-sparse plan's route prepare follows it as a programmatic dependent
    launch that acquires this grid before exiting, and attention follows the
    route prepare in stream order.
    """

    def __init__(
        self,
        *,
        cfg: FmhaDecodeConfig,
        k_image: _SageKScaleImageLayout,
        batch_size: int,
        num_kv_heads: int,
    ) -> None:
        self.cfg = cfg
        self.k_image = k_image
        self.num_kv_heads = num_kv_heads
        num_sequences = batch_size * num_kv_heads
        _, k_sequence_words = k_image.shape(num_sequences)
        _, q_sequence_words = cfg.sage_q_scale_image.shape(num_sequences)
        self.num_k_pieces = num_sequences * k_sequence_words // _SAGE_IMAGE_PIECE_WORDS
        self.num_pieces = (
            self.num_k_pieces
            + num_sequences * q_sequence_words // _SAGE_IMAGE_PIECE_WORDS
        )

    @cute.jit
    def __call__(
        self,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        k_summary_scale: cute.Tensor | None,
        q_scale_image: cute.Tensor,
        k_scale_image: cute.Tensor,
        stream: cuda_drv.CUstream,
    ) -> None:
        self.kernel(
            q_scale, k_scale, k_summary_scale, q_scale_image, k_scale_image
        ).launch(
            grid=[ceil_div(self.num_pieces, _THREADS_PER_CTA), 1, 1],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        k_summary_scale: cute.Tensor | None,
        q_scale_image: cute.Tensor,
        k_scale_image: cute.Tensor,
    ) -> None:
        """Write this thread's piece of the K image or the Q image, if any."""

        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        piece_idx = block_idx * cutlass.Int32(_THREADS_PER_CTA) + thread_idx
        if piece_idx < cutlass.Int32(self.num_k_pieces):
            _write_k_image_piece(
                k_scale,
                k_summary_scale,
                k_scale_image,
                piece_idx,
                self.k_image,
                self.num_kv_heads,
            )
        elif piece_idx < cutlass.Int32(self.num_pieces):
            _write_q_image_piece(
                q_scale,
                q_scale_image,
                piece_idx - cutlass.Int32(self.num_k_pieces),
                self.cfg,
                self.num_kv_heads,
            )


__all__ = ["_PrepareSageScaleImages"]
