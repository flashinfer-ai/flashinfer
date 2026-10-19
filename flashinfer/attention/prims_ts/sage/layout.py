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

"""Word layout of the FP32 scale buffer a Sage plan prepares before attention.

The Sage prepare (``prepare.py``) gathers a plan's flat scale tensors into one
buffer in the order the attention kernels read them; ``_SageScaleLayout``
describes that buffer and its per-sequence sections. Both the decode and the
context kernels read it; each kernel fixes the parameters of its own reads
(the Q tile rows, the K tile atoms).
"""

from __future__ import annotations

from dataclasses import dataclass

from flashinfer.utils import ceil_div, round_up

_FLOAT4_WORDS = 4
# A Sage K-scale atom is one Keeps layout atom of 64 K tokens, read one word
# per 32-token score fragment at least, so it holds at least two words. The
# prepared scales are written and copied in float4 units, the width of one
# ``cp.async``.
_SAGE_ATOM_TOKENS = 64
_SAGE_ATOM_MIN_WORDS = 2


def _validate_i32_address(value: int, name: str) -> int:
    """Reject a buffer extent device Int32 addressing cannot hold."""

    if value > 2**31 - 1:
        raise OverflowError(f"{name} must fit in signed int32")
    return value


@dataclass(frozen=True)
class _SageKSection:
    """The ``sfK`` words of one route kind: one atom per 64 tokens.

    An atom covers 64 tokens (two K32 score fragments) and holds
    ``words_per_atom = max(2, 64 / block_size)`` scales, word ``w`` being the
    scale of the atom's token ``w * tokens_per_word``, padded with zeros to
    whole float4s. Scales come from the flat source tensor of one scale per
    ``source_block_size`` tokens; a coarser source repeats its value, and a
    token past ``length`` takes the last scale. The atoms are padded to whole
    groups of ``tile_atoms``, so a reader that addresses whole K tiles never
    runs past the section; a pad atom repeats the last scale.
    """

    length: int
    block_size: int
    source_block_size: int
    word_offset: int
    tile_atoms: int = 1

    @property
    def words_per_atom(self) -> int:
        """Return the words of one atom that hold scales: one per block, at least two."""

        return max(_SAGE_ATOM_MIN_WORDS, _SAGE_ATOM_TOKENS // self.block_size)

    @property
    def tokens_per_word(self) -> int:
        """Return the tokens from one word's token to the next word's."""

        return _SAGE_ATOM_TOKENS // self.words_per_atom

    @property
    def atom_words(self) -> int:
        """Return the words of one atom: ``words_per_atom`` padded to whole float4s."""

        return round_up(self.words_per_atom, _FLOAT4_WORDS)

    @property
    def atom_float4s(self) -> int:
        """Return the float4s of one atom."""

        return self.atom_words // _FLOAT4_WORDS

    @property
    def atoms(self) -> int:
        """Return the section's 64-token atoms, padded to whole ``tile_atoms`` groups."""

        return round_up(ceil_div(self.length, _SAGE_ATOM_TOKENS), self.tile_atoms)

    @property
    def words(self) -> int:
        """Return the words of the section's atoms."""

        return self.atoms * self.atom_words


@dataclass(frozen=True)
class _SageQSection:
    """The ``sfQ`` words of a sequence's Q tiles: one word per row of each tile.

    A Q tile is what one softmax instance reads: ``rows_per_tile`` rows, row
    ``r`` of tile ``t`` at word ``word_offset + t * rows_per_tile + r``, in
    the order the softmax lanes read them. With ``heads_interleaved`` (the
    decode kernel's grouped tile) a tile holds ``tokens_per_tile`` tokens of
    each of the ``heads_q_per_kv`` Q heads of its KV head, row ``r`` being
    token ``r // heads_q_per_kv`` of local head ``r % heads_q_per_kv``.
    Otherwise (the context kernel's tile) a tile holds ``tokens_per_tile``
    tokens of one Q head: tile ``h * token_tiles + i`` is token range ``i`` of
    local head ``h`` and row ``r`` its token ``r``. The word is that head's
    scale of the token's ``q_block_size`` block; a row whose token lies past
    the sequence's end takes the last token's, so a masked row keeps a
    finite positive scale.
    """

    seq_len_q: int
    q_block_size: int
    heads_q_per_kv: int
    tokens_per_tile: int
    word_offset: int
    heads_interleaved: bool

    def __post_init__(self) -> None:
        if self.tokens_per_tile < 1 or self.rows_per_tile % _FLOAT4_WORDS:
            raise ValueError("sfQ tiles hold whole float4s of at least one token")

    @property
    def token_tiles(self) -> int:
        """Return the ``tokens_per_tile`` token ranges of the sequence, the last possibly partial."""

        return ceil_div(self.seq_len_q, self.tokens_per_tile)

    @property
    def rows_per_tile(self) -> int:
        """Return the rows of one tile: its tokens of each head it holds."""

        heads = self.heads_q_per_kv if self.heads_interleaved else 1
        return self.tokens_per_tile * heads

    @property
    def tiles(self) -> int:
        """Return the Q tiles of one sequence."""

        heads = 1 if self.heads_interleaved else self.heads_q_per_kv
        return heads * self.token_tiles

    @property
    def words(self) -> int:
        """Return the words of the section: ``rows_per_tile`` per tile."""

        return self.tiles * self.rows_per_tile

    def tile_index(self, local_head, token_tile):
        """Return the tile holding token range ``token_tile`` of local Q head ``local_head``.

        The arithmetic serves host integers and device ``Int32`` values.
        """

        if self.heads_interleaved:
            return token_tile
        return local_head * self.token_tiles + token_tile

    def row_token_and_head(self, tile, row):
        """Return the token and the local Q head of row ``row`` of tile ``tile``.

        The token is not clamped to the sequence. The arithmetic serves host
        integers, tensors and device ``Int32`` values.
        """

        if self.heads_interleaved:
            token_in_tile = row // self.heads_q_per_kv
            local_head = row - token_in_tile * self.heads_q_per_kv
            return tile * self.tokens_per_tile + token_in_tile, local_head
        local_head = tile // self.token_tiles
        token_tile = tile - local_head * self.token_tiles
        return token_tile * self.tokens_per_tile + row, local_head


@dataclass(frozen=True)
class _SageVSection:
    """The ``sfV`` row and, with a V mean, the ``v_mean`` row of one sequence.

    Row ``r`` holds channel ``c`` at word ``word_offset + r * headdim + c``
    of the sequence, copied from the sequence's rows of the ``[B, Hkv, D]``
    tensors as they are.
    """

    headdim: int
    has_mean: bool
    word_offset: int

    def __post_init__(self) -> None:
        if self.headdim % _FLOAT4_WORDS:
            raise ValueError("sfV rows hold whole float4s")

    @property
    def rows(self) -> int:
        """Return the rows of one head: the scales, then the means."""

        return 2 if self.has_mean else 1

    @property
    def mean_word_offset(self) -> int:
        """Return the first word of the mean row within the section."""

        return self.headdim

    @property
    def words(self) -> int:
        """Return the words of the section's rows."""

        return self.rows * self.headdim


@dataclass(frozen=True)
class _SageScaleLayout:
    """Word layout of the FP32 scale buffer a Sage plan prepares before attention.

    The buffer is one block of ``sequence_words`` words per sequence ``s = b
    * Hkv + h`` (KV head ``h`` of batch entry ``b``), each holding four
    sections::

        | k_tokens | summaries (proxy plans only) | q_rows | v_rows |

    so any scale of a sequence is ``s * sequence_words + section.word_offset
    + offset within the section``.

    ``k_tokens`` and ``summaries`` are ``_SageKSection`` atoms of the block
    size each route kind reads (``FmhaDecodeConfig.sage_k_scale_block_size``):
    16-token K blocks give four words per atom (tokens 0, 16, 32, 48),
    one-token scales 64, blocks of 64 tokens or more the two-word floor (one
    per fragment) padded to a float4. ``q_rows`` is a ``_SageQSection`` and
    ``v_rows`` a ``_SageVSection``. Every section starts on a float4 and K
    atoms are whole float4s because the attention load warp copies them with
    16-byte ``cp.async``.
    """

    seq_len_kv: int
    k_block_size: int
    # Summary sequence length of a proxy plan, else zero.
    num_summaries: int
    k_summary_block_size: int
    k_tokens_block_size: int
    summaries_block_size: int
    seq_len_q: int
    q_block_size: int
    heads_q_per_kv: int
    # Q tokens per Q tile, per head, and whether a tile interleaves the heads
    # of its KV head (``_SageQSection``).
    q_tokens_per_tile: int
    q_heads_interleaved: bool
    headdim: int
    v_mean: bool
    # Atoms per K tile of the reader; the K-token atoms are padded to whole
    # tiles (``_SageKSection.tile_atoms``).
    k_tile_atoms: int = 1

    @property
    def k_tokens(self) -> _SageKSection:
        """Return the K-token section, which starts a sequence's words."""

        return _SageKSection(
            length=self.seq_len_kv,
            block_size=self.k_tokens_block_size,
            source_block_size=self.k_block_size,
            word_offset=0,
            tile_atoms=self.k_tile_atoms,
        )

    @property
    def summaries(self) -> _SageKSection | None:
        """Return a proxy plan's summary section, which follows the K tokens."""

        if self.num_summaries == 0:
            return None
        return _SageKSection(
            length=self.num_summaries,
            block_size=self.summaries_block_size,
            source_block_size=self.k_summary_block_size,
            word_offset=self.k_tokens.words,
        )

    @property
    def k_sections(self) -> tuple[_SageKSection, ...]:
        """Return the K sections in order, paired with ``k_scale`` then ``k_summary_scale``."""

        summaries = self.summaries
        return (self.k_tokens,) if summaries is None else (self.k_tokens, summaries)

    @property
    def q_rows(self) -> _SageQSection:
        """Return the Q section, which follows the K sections."""

        return _SageQSection(
            seq_len_q=self.seq_len_q,
            q_block_size=self.q_block_size,
            heads_q_per_kv=self.heads_q_per_kv,
            tokens_per_tile=self.q_tokens_per_tile,
            word_offset=sum(section.words for section in self.k_sections),
            heads_interleaved=self.q_heads_interleaved,
        )

    @property
    def v_rows(self) -> _SageVSection:
        """Return the V section, which follows the Q rows."""

        q_rows = self.q_rows
        return _SageVSection(
            headdim=self.headdim,
            has_mean=self.v_mean,
            word_offset=q_rows.word_offset + q_rows.words,
        )

    @property
    def sequence_words(self) -> int:
        """Return the words of one sequence: K sections, Q rows, then V rows."""

        v_rows = self.v_rows
        return v_rows.word_offset + v_rows.words

    def total_words(self, num_sequences: int) -> int:
        """Return the words of the buffer for ``batch * Hkv`` sequences."""

        return _validate_i32_address(
            num_sequences * self.sequence_words, "sage_scales_words"
        )
