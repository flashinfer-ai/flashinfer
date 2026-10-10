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

"""Shared layouts of the route records and Sage K-scale image prepared before attention."""

from dataclasses import dataclass

from flashinfer.utils import ceil_div, round_up

from .common import _SIGNED_INT32_MAX, _block_sparse_kv_atom_size

_SECTION_ALIGNMENT_WORDS = 4
_PREPARED_ROUTE_IS_FULL_FLAG = 1 << 0
_PREPARED_ROUTE_IS_PROXY_FLAG = 1 << 1
_SUPPORTED_KV_ROUTE_SIZES = (128, 256)
_SUPPORTED_PAGED_KV_PAGE_SIZES = (16, 32, 64, 128)
# A Sage route atom is one Keeps layout atom: two K32 score fragments. The
# attention load warp copies the image in 16-byte pieces of four words.
_SAGE_IMAGE_ATOM_TOKENS = 64
_SAGE_IMAGE_FRAGMENT_TOKENS = 32
_SAGE_IMAGE_PIECE_WORDS = 4


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
class _SageKScaleImageKind:
    """One route kind's chunks within a sequence's ``sfK`` image words.

    The kind's ``length`` tokens (K tokens, or block summaries) take one
    chunk per 64-token atom from word ``word_offset`` of the sequence. The
    chunk words hold one scale per ``block_size`` tokens, read from the
    kind's flat source tensor of one scale per ``source_block_size`` tokens.
    """

    length: int
    block_size: int
    source_block_size: int
    word_offset: int

    @property
    def groups(self) -> int:
        """Return the scale groups of one K32 fragment, at least one."""

        return max(1, _SAGE_IMAGE_FRAGMENT_TOKENS // self.block_size)

    @property
    def group_tokens(self) -> int:
        """Return the tokens from one group's token to the next group's."""

        return _SAGE_IMAGE_FRAGMENT_TOKENS // self.groups

    @property
    def used_words(self) -> int:
        """Return the words of one chunk that hold scales."""

        return _SAGE_IMAGE_ATOM_TOKENS // _SAGE_IMAGE_FRAGMENT_TOKENS * self.groups

    @property
    def chunk_words(self) -> int:
        """Return the words of one chunk: ``used_words`` padded to whole pieces."""

        return round_up(self.used_words, _SAGE_IMAGE_PIECE_WORDS)

    @property
    def atoms(self) -> int:
        """Return the kind's 64-token atoms, the last one possibly partial."""

        return ceil_div(self.length, _SAGE_IMAGE_ATOM_TOKENS)

    @property
    def words(self) -> int:
        """Return the words of the kind's chunks."""

        return self.atoms * self.chunk_words


@dataclass(frozen=True)
class _SageKScaleImageLayout:
    """Word layout of the ``sfK`` image a Sage plan prepares before attention.

    The image is ``image[b * Hkv + h][kind][atom][word]`` in FP32 words: KV
    head ``h`` of sequence ``b`` owns ``sequence_words`` consecutive words
    holding its K-token chunks (the exact kind) and then, for a proxy plan,
    its block-summary chunks (the summary kind). A kind has one chunk per
    64-token atom of its tokens, padded to whole 16-byte pieces because the
    attention load warp copies the image in pieces. Word ``word = fragment *
    groups + group`` of a chunk is the scale of the atom's token ``atom * 64 +
    word * (32 / groups)``: scale group ``group`` of the atom's K32 fragment
    ``fragment``, in the order the softmax reads them; words from ``2 *
    groups`` on pad the chunk and are zero. A token past the kind's length
    takes the last scale, so masked scores keep a finite positive scale.

    ``groups`` is ``32 / block_size`` of the block size the kind reads
    (``exact_block_size``, ``summary_block_size``, resolved by
    ``FmhaDecodeConfig.sage_k_scale_block_size``), at least one. With
    16-token K blocks a chunk has two groups per fragment and four words, the
    ``sfK`` of the atom's tokens 0, 16, 32 and 48; with one-token K scales it
    has 64 words, one per token; with 64-token or coarser blocks it has two
    words, the scale of tokens 0 and 32, padded to one piece. A kind whose
    source scales (``k_block_size``, ``k_summary_block_size``) are coarser
    than the block size it reads repeats them.
    """

    seq_len_kv: int
    k_block_size: int
    # Summary sequence length of a proxy plan, else zero.
    num_summaries: int
    k_summary_block_size: int
    exact_block_size: int
    summary_block_size: int

    @property
    def exact(self) -> _SageKScaleImageKind:
        """Return the K-token kind, which starts a sequence's words."""

        return _SageKScaleImageKind(
            length=self.seq_len_kv,
            block_size=self.exact_block_size,
            source_block_size=self.k_block_size,
            word_offset=0,
        )

    @property
    def summary(self) -> _SageKScaleImageKind | None:
        """Return a proxy plan's summary kind, which follows the K-token chunks."""

        if self.num_summaries == 0:
            return None
        return _SageKScaleImageKind(
            length=self.num_summaries,
            block_size=self.summary_block_size,
            source_block_size=self.k_summary_block_size,
            word_offset=self.exact.words,
        )

    @property
    def kinds(self) -> tuple[_SageKScaleImageKind, ...]:
        """Return the kinds in image order, paired with ``k_scale`` then ``k_summary_scale``."""

        summary = self.summary
        return (self.exact,) if summary is None else (self.exact, summary)

    @property
    def sequence_words(self) -> int:
        """Return the words of one sequence and KV head: K tokens, then summaries."""

        return sum(kind.words for kind in self.kinds)

    def shape(self, num_sequences: int) -> tuple[int, int]:
        """Return the image shape ``[num_sequences, sequence_words]`` for ``batch * Hkv`` sequences."""

        _validate_i32_address(
            num_sequences * self.sequence_words,
            "sage_k_scale_image_words",
        )
        return num_sequences, self.sequence_words


__all__ = [
    "_PREPARED_ROUTE_IS_FULL_FLAG",
    "_PREPARED_ROUTE_IS_PROXY_FLAG",
    "_BlockSparseRouteLayout",
    "_SageKScaleImageKind",
    "_SageKScaleImageLayout",
]
