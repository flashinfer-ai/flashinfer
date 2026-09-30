"""Compile geometry and prepared-tile workspace for delta-rule prefill."""

BACKEND = "cutedsl"
V_SPLIT_CHOICES = (16, 32, 64, 128)
K_SPLIT_CHOICES = (1, 2, 4)
STAGE_CHOICES = (2, 3, 4)
CHUNK_TOKENS = 16


class WorkspaceRecord:
    """Byte layout of one prepared (tile, head) record in the workspace ring.

    The recurrence kernel copies ``[0, HEAD_BYTES)`` into a pipeline stage
    with one bulk copy; the stage then holds this CTA's value rows from
    offset ``V`` as ``v_split // 16`` groups of ``[16 tokens x 16 values]``,
    copied straight from the value tensor. The operand tiles are stored in
    the swizzled 16-byte-chunk order the consumer's ldmatrix reads.
    """

    Q_TILDE = 0
    K_TILDE = 4096
    K_R = 8192
    INV = 12288
    MQK = 12800
    LAMBDA_C = 13312
    BETA = 13824
    SUMMARY_FINITE = 13888
    HEAD_BYTES = 13952
    BYTES = 14080
    V = 14080


WORKSPACE_RECORD_BYTES = WorkspaceRecord.BYTES
# Prepared-tile bytes one window may occupy; two windows stay L2 resident.
WINDOW_BYTES_BUDGET = 36 << 20
SM12X_SHARED_MEMORY_LIMIT = 99 * 1024


def recurrence_shared_bytes(v_split: int, k_split: int, stages: int, window_tiles: int,
                            *, max_sequence_tiles: int = 0, summary_mode: int = 0,
                            reuse_value_buffer: bool = True) -> int:
    """Kernel-allocated bytes for band positions, pipeline, reductions, and barriers."""
    def align(value, multiple):
        return -(-value // multiple) * multiple

    band_tiles = min(window_tiles, max_sequence_tiles or window_tiles)
    record_skip = WorkspaceRecord.K_TILDE if summary_mode in (1, 2, 3) else 0
    cursor = align(4 * (band_tiles + 1), 128)
    cursor += stages * (WorkspaceRecord.V - record_skip + 32 * v_split)
    if not reuse_value_buffer:
        cursor = align(cursor, 128) + 32 * v_split
    reduction = 2 * (v_split // 16) * k_split * 256 * 4 if k_split > 1 else 16
    cursor = align(cursor, 128) + reduction
    cursor = align(cursor, 8) + 16 * stages
    return cursor + (4 if summary_mode in (2, 3, 5) and k_split == 1 else 0)


def validate_shared_memory(v_split: int, k_split: int, stages: int, window_tiles: int,
                           *, max_sequence_tiles: int = 0, summary_mode: int = 0,
                           reuse_value_buffer: bool = True) -> None:
    required = recurrence_shared_bytes(v_split, k_split, stages, window_tiles,
                                      max_sequence_tiles=max_sequence_tiles, summary_mode=summary_mode,
                                      reuse_value_buffer=reuse_value_buffer)
    if required > SM12X_SHARED_MEMORY_LIMIT:
        raise ValueError(
            f"prefill recurrence requires {required} shared-memory bytes; "
            f"SM12x permits {SM12X_SHARED_MEMORY_LIMIT}"
        )


def tiles_capacity(max_tokens: int, max_seqs: int) -> int:
    """Upper bound on packed chunk tiles: one partial tile per sequence."""
    return -(-int(max_tokens) // CHUNK_TOKENS) + int(max_seqs)


def default_window_tiles(heads: int, max_tokens: int, max_seqs: int) -> int:
    """Tiles per pipeline window so a window's prepared tiles fit the L2 budget."""
    per_row = int(heads) * WORKSPACE_RECORD_BYTES
    return max(1, min(tiles_capacity(max_tokens, max_seqs), WINDOW_BYTES_BUDGET // per_row))
