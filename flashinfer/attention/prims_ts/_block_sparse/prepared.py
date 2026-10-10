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

"""Shared layouts of the route records and Sage scales prepared before attention."""

from dataclasses import dataclass

from flashinfer.utils import ceil_div, round_up

from .common import _SIGNED_INT32_MAX, _block_sparse_kv_atom_size

_SECTION_ALIGNMENT_WORDS = 4
_PREPARED_ROUTE_IS_FULL_FLAG = 1 << 0
_PREPARED_ROUTE_IS_PROXY_FLAG = 1 << 1
_SUPPORTED_KV_ROUTE_SIZES = (128, 256)
_SUPPORTED_PAGED_KV_PAGE_SIZES = (16, 32, 64, 128)
# A Sage K-scale atom is one Keeps layout atom of 64 K tokens, read one word
# per 32-token score fragment at least, so it holds at least two words. The
# prepared scales are written and copied in float4 units, the width of one
# ``cp.async``.
_SAGE_ATOM_TOKENS = 64
_SAGE_ATOM_MIN_WORDS = 2
_FLOAT4_WORDS = 4


def _validate_int(value: object, name: str, *, allow_zero: bool) -> int:
    """Validate a host layout extent while rejecting ``bool`` explicitly."""

    requirement = "non-negative" if allow_zero else "positive"
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a {requirement} Python integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {requirement}")
    return value


def _validate_i32_address(value: int, name: str) -> int:
    """Reject an allocation extent or section address device Int32 cannot hold."""

    if value > _SIGNED_INT32_MAX:
        raise OverflowError(f"{name} must fit in signed int32")
    return value


def _build_route_workspace_geometry(
    *,
    route_metadata_words: int,
    route_metadata_capacity: int,
    num_rows: int,
) -> tuple[int, int, int]:
    """Return the shared aligned route stride, base, and workspace extent."""

    route_metadata_stride_words = round_up(
        route_metadata_words,
        _SECTION_ALIGNMENT_WORDS,
    )
    _validate_i32_address(
        num_rows + 1,
        "row_route_offsets_length",
    )
    route_metadata_base_word_offset = _validate_i32_address(
        round_up(num_rows, _SECTION_ALIGNMENT_WORDS),
        "route_metadata_base_word_offset",
    )
    workspace_size_words = _validate_i32_address(
        route_metadata_base_word_offset
        + route_metadata_capacity * route_metadata_stride_words,
        "workspace_size_words",
    )
    return (
        route_metadata_stride_words,
        route_metadata_base_word_offset,
        workspace_size_words,
    )


@dataclass(frozen=True)
class _BlockSparseRouteLayout:
    """Immutable word layout for compact, prepared sparse routes.

    A separate immutable tensor holds the plan-generated ``num_rows + 1``
    CSR-style row offsets. Each offset is a route ordinal into the metadata
    section, not a workspace word offset. This layout describes only mutable
    run scratch: ``num_rows`` route counts from word zero, followed by
    ``route_metadata_capacity`` fixed-stride route metadata on a four-word
    (16-byte) boundary. Every plan assigns uniform-capacity row slices so
    caller-owned BSR boundaries may change between runs.

    Each route's metadata stores logical KV-token atom origins, optional
    physical page IDs, one atom-valid-mask word, one route-flags word, and
    optional token-mask words. ``page_size is None`` selects the contiguous
    record; otherwise the paged record adds one page-ID word per logical
    origin. Logical origins remain independent of the K/V storage locator used
    by the attention load path. Exact routes address raw-token origins, while
    proxy routes address summary-token origins and set
    ``_PREPARED_ROUTE_IS_PROXY_FLAG`` (bit 1). An invalid logical origin is
    encoded as ``-1``. Bit ``i`` of the atom-valid mask corresponds to logical
    origin ``i``. ``_PREPARED_ROUTE_IS_FULL_FLAG`` (bit 0) states that the
    route is both structurally full and, when token-mask bits are present,
    mask-full.
    """

    # Store semantic inputs plus the three validated allocation values. All
    # remaining offsets and extents are derived from this route geometry.
    # Number of logical KV tokens represented by one prepared route record.
    kv_route_size: int
    # Smallest independently addressable KV fragment represented by metadata.
    atom_size: int
    # Whether each route's metadata carries per-token validity words.
    has_token_bits: bool
    # Flattened (batch, KV head, Q-block row) count.
    num_rows: int
    # route_workspace = [row route counts | padding | route metadata].
    # Aligned Int32-word distance between adjacent routes' metadata.
    route_metadata_stride_words: int
    # Base offset from route_workspace[0] to the first route's metadata.
    route_metadata_base_word_offset: int
    # Total mutable workspace extent in Int32 words.
    workspace_size_words: int
    # Paged-KV token capacity, or None for contiguous K/V storage.
    page_size: int | None = None

    @staticmethod
    def create(
        *,
        kv_route_size: int,
        kv_block_size: int,
        has_token_bits: bool,
        route_metadata_capacity: int,
        num_rows: int,
        page_size: int | None = None,
    ) -> "_BlockSparseRouteLayout":
        """Build aligned workspace-section and per-route metadata geometry."""

        kv_route_size = _validate_int(
            kv_route_size,
            "kv_route_size",
            allow_zero=False,
        )
        if kv_route_size not in _SUPPORTED_KV_ROUTE_SIZES:
            raise ValueError("kv_route_size must be 128 or 256")
        atom_size = _block_sparse_kv_atom_size(kv_block_size)
        if page_size is not None:
            page_size = _validate_int(page_size, "page_size", allow_zero=False)
            if atom_size > page_size:
                raise ValueError("atom_size must not exceed page_size")
            if page_size % atom_size != 0:
                raise ValueError("page_size must be divisible by atom_size")
            if page_size not in _SUPPORTED_PAGED_KV_PAGE_SIZES:
                raise ValueError("page_size must be 16, 32, 64, or 128")
        logical_origins_per_route = kv_route_size // atom_size
        if not isinstance(has_token_bits, bool):
            raise TypeError("has_token_bits must be a bool")
        route_metadata_capacity = _validate_int(
            route_metadata_capacity,
            "route_metadata_capacity",
            allow_zero=True,
        )
        num_rows = _validate_int(num_rows, "num_rows", allow_zero=False)

        token_words_per_route = kv_route_size // 32
        (
            route_metadata_stride_words,
            route_metadata_base_word_offset,
            workspace_size_words,
        ) = _build_route_workspace_geometry(
            route_metadata_words=(
                logical_origins_per_route
                + (logical_origins_per_route if page_size is not None else 0)
                + 2
                + (token_words_per_route if has_token_bits else 0)
            ),
            route_metadata_capacity=route_metadata_capacity,
            num_rows=num_rows,
        )

        return _BlockSparseRouteLayout(
            kv_route_size=kv_route_size,
            atom_size=atom_size,
            has_token_bits=has_token_bits,
            num_rows=num_rows,
            route_metadata_stride_words=route_metadata_stride_words,
            route_metadata_base_word_offset=route_metadata_base_word_offset,
            workspace_size_words=workspace_size_words,
            page_size=page_size,
        )

    @property
    def is_paged(self) -> bool:
        """Whether each route carries physical-page-ID locator words."""

        return self.page_size is not None

    @property
    def paged_page_size(self) -> int:
        """Return the paged token capacity, failing on contiguous layouts."""

        if self.page_size is None:
            raise RuntimeError("paged page size requested from contiguous layout")
        return self.page_size

    @property
    def logical_origins_per_route(self) -> int:
        """Number of logical KV atom origins stored in one route."""

        return self.kv_route_size // self.atom_size

    @property
    def token_words_per_route(self) -> int:
        """Number of 32-token validity words covered by one route."""

        return self.kv_route_size // 32

    @property
    def physical_page_ids_word_offset(self) -> int:
        """First physical-page-ID word in a paged record's locator section."""

        if not self.is_paged:
            raise RuntimeError("page-ID offset requested from contiguous layout")
        return self.logical_origins_per_route

    @property
    def atom_valid_mask_word_offset(self) -> int:
        """Word holding one validity bit for each route KV atom."""

        locator_words = self.logical_origins_per_route if self.is_paged else 0
        return self.logical_origins_per_route + locator_words

    @property
    def route_flags_word_offset(self) -> int:
        """Word holding route-wide flags such as ``ROUTE_IS_FULL``."""

        return self.atom_valid_mask_word_offset + 1

    @property
    def token_words_word_offset(self) -> int | None:
        """First per-token validity word, or ``None`` for unmasked routes."""

        return self.route_flags_word_offset + 1 if self.has_token_bits else None

    @property
    def uses_one_warp_transport(self) -> bool:
        """Whether this layout uses the continuous one-warp transport."""

        token_words_word_offset = self.token_words_word_offset
        return (
            not self.is_paged
            and token_words_word_offset is not None
            and token_words_word_offset + self.token_words_per_route <= 32
        )

    @property
    def route_metadata_capacity(self) -> int:
        """Number of routes whose metadata fits in the mutable workspace."""

        return (
            self.workspace_size_words - self.route_metadata_base_word_offset
        ) // self.route_metadata_stride_words


@dataclass(frozen=True)
class _SageKSection:
    """The ``sfK`` words of one route kind: one atom per 64 tokens.

    An atom covers 64 tokens (two K32 score fragments) and holds
    ``words_per_atom = max(2, 64 / block_size)`` scales, word ``w`` being the
    scale of the atom's token ``w * tokens_per_word``, padded with zeros to
    whole float4s. Scales come from the flat source tensor of one scale per
    ``source_block_size`` tokens; a coarser source repeats its value, and a
    token past ``length`` takes the last scale.
    """

    length: int
    block_size: int
    source_block_size: int
    word_offset: int

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
        """Return the section's 64-token atoms, the last one possibly partial."""

        return ceil_div(self.length, _SAGE_ATOM_TOKENS)

    @property
    def words(self) -> int:
        """Return the words of the section's atoms."""

        return self.atoms * self.atom_words


@dataclass(frozen=True)
class _SageQSection:
    """The ``sfQ`` words of a sequence's Q tiles: one word per tile row.

    Tile ``t`` holds ``tile_size_q`` words from word ``word_offset + t *
    tile_size_q``, in the order the softmax lanes read them: row ``r`` is
    token ``t * tokens_per_tile + r // heads_q_per_kv`` of local Q head ``r %
    heads_q_per_kv``, and its word is that head's scale of the token's
    ``q_block_size`` block. Rows past the sequence's last token repeat the
    last valid word, so a masked row keeps a finite positive scale.
    """

    seq_len_q: int
    q_block_size: int
    heads_q_per_kv: int
    tile_size_q: int
    word_offset: int

    def __post_init__(self) -> None:
        if self.tile_size_q % _FLOAT4_WORDS or self.heads_q_per_kv > self.tile_size_q:
            raise ValueError("sfQ tiles hold whole float4s of at least one token")

    @property
    def tokens_per_tile(self) -> int:
        """Return the complete Q tokens of one tile."""

        return self.tile_size_q // self.heads_q_per_kv

    @property
    def tiles(self) -> int:
        """Return the Q tiles of one sequence, the last one possibly partial."""

        return ceil_div(self.seq_len_q, self.tokens_per_tile)

    def valid_rows(self, tile_idx: int) -> int:
        """Return the rows of tile ``tile_idx`` that hold a token of the sequence."""

        remaining = self.seq_len_q - tile_idx * self.tokens_per_tile
        return min(remaining, self.tokens_per_tile) * self.heads_q_per_kv

    @property
    def words(self) -> int:
        """Return the words of the section: ``tile_size_q`` per tile."""

        return self.tiles * self.tile_size_q


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
    tile_size_q: int
    headdim: int
    v_mean: bool

    @property
    def k_tokens(self) -> _SageKSection:
        """Return the K-token section, which starts a sequence's words."""

        return _SageKSection(
            length=self.seq_len_kv,
            block_size=self.k_tokens_block_size,
            source_block_size=self.k_block_size,
            word_offset=0,
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
            tile_size_q=self.tile_size_q,
            word_offset=sum(section.words for section in self.k_sections),
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


__all__ = [
    "_PREPARED_ROUTE_IS_FULL_FLAG",
    "_PREPARED_ROUTE_IS_PROXY_FLAG",
    "_BlockSparseRouteLayout",
    "_SageKSection",
    "_SageQSection",
    "_SageScaleLayout",
    "_SageVSection",
]
