"""Compressed MLA kernel planning helpers shared by runtime and scratch plans."""

from __future__ import annotations

from b12x.attention._shared.workspace import (
    SparseMLASplitDecodeConfig,
)


_COMPRESSED_SPARSE_MLA_DECODE_SPLIT_CHUNK_SIZE = 12
# vLLM captures MTP decode graphs with rows = requests * (1 + draft tokens).
# Keep those graph-capacity rows on the decode split contract; switching to the
# batched split contract at 66+ rows can poison smaller decode graph replays.
_COMPRESSED_SPARSE_MLA_DECODE_SPLIT_MAX_ROWS = 256
_COMPRESSED_SPARSE_MLA_DECODE_WIDE_CHUNK_SIZE = 64
_COMPRESSED_SPARSE_MLA_BATCHED_SPLIT_CHUNK_SIZE = 1024
_COMPRESSED_SPARSE_MLA_SPLIT_MAX_CHUNKS = 256


def compressed_sparse_mla_split_config_for_contract(
    *,
    rows: int,
    width: int,
    max_chunks: int | None = None,
    decode_row_capacity: int | None = None,
) -> SparseMLASplitDecodeConfig:
    rows = max(int(rows), 1)
    width = max(int(width), 1)
    decode_split_max_rows = _COMPRESSED_SPARSE_MLA_DECODE_SPLIT_MAX_ROWS
    if decode_row_capacity is not None:
        decode_row_capacity = int(decode_row_capacity)
        if decode_row_capacity <= 0:
            raise ValueError(
                f"decode_row_capacity must be positive, got {decode_row_capacity}"
            )
        decode_split_max_rows = max(decode_split_max_rows, decode_row_capacity)
    chunk_limit = _COMPRESSED_SPARSE_MLA_SPLIT_MAX_CHUNKS
    if max_chunks is not None:
        chunk_limit = max(1, min(int(max_chunks), chunk_limit))

    decode_chunks = (
        width + _COMPRESSED_SPARSE_MLA_DECODE_SPLIT_CHUNK_SIZE - 1
    ) // _COMPRESSED_SPARSE_MLA_DECODE_SPLIT_CHUNK_SIZE
    if rows <= decode_split_max_rows and decode_chunks <= chunk_limit:
        return SparseMLASplitDecodeConfig(
            chunk_size=_COMPRESSED_SPARSE_MLA_DECODE_SPLIT_CHUNK_SIZE,
            num_chunks=decode_chunks,
        )

    wide_decode_chunks = (
        width + _COMPRESSED_SPARSE_MLA_DECODE_WIDE_CHUNK_SIZE - 1
    ) // _COMPRESSED_SPARSE_MLA_DECODE_WIDE_CHUNK_SIZE
    if rows <= decode_split_max_rows and wide_decode_chunks <= chunk_limit:
        return SparseMLASplitDecodeConfig(
            chunk_size=_COMPRESSED_SPARSE_MLA_DECODE_WIDE_CHUNK_SIZE,
            num_chunks=wide_decode_chunks,
        )

    chunks = (
        width + _COMPRESSED_SPARSE_MLA_BATCHED_SPLIT_CHUNK_SIZE - 1
    ) // _COMPRESSED_SPARSE_MLA_BATCHED_SPLIT_CHUNK_SIZE
    if chunks <= chunk_limit:
        return SparseMLASplitDecodeConfig(
            chunk_size=_COMPRESSED_SPARSE_MLA_BATCHED_SPLIT_CHUNK_SIZE,
            num_chunks=chunks,
        )

    chunk_size = (width + chunk_limit - 1) // chunk_limit
    return SparseMLASplitDecodeConfig(chunk_size=chunk_size, num_chunks=chunk_limit)


def compressed_sparse_mla_split_chunks_for_contract(
    *,
    rows: int,
    width: int,
    max_chunks: int | None = None,
    decode_row_capacity: int | None = None,
) -> int:
    return compressed_sparse_mla_split_config_for_contract(
        rows=rows,
        width=width,
        max_chunks=max_chunks,
        decode_row_capacity=decode_row_capacity,
    ).num_chunks


__all__ = [
    "compressed_sparse_mla_split_chunks_for_contract",
    "compressed_sparse_mla_split_config_for_contract",
]
