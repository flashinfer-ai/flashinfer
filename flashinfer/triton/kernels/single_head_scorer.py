import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


@triton.jit
def single_head_scorer_kernel(
    query_ptr,
    query_stride,
    key_ptr,
    key_page_stride,
    key_row_stride,
    block_table_ptr,
    block_table_stride,
    context_lengths_ptr,
    requests_ptr,
    scores_ptr,
    scores_stride,
    ROWS: tl.constexpr,
):
    """Score one 64-key page of a group's request against the group's ROWS
    BF16 query rows, padded to 16 rows for the MMA."""
    group, page = tl.program_id(0), tl.program_id(1)
    length = tl.load(context_lengths_ptr + group)
    if page * 64 >= length:
        return
    request = tl.load(requests_ptr + group).to(tl.int64)
    block = tl.load(block_table_ptr + request * block_table_stride + page)
    row, key, dim = tl.arange(0, 16), tl.arange(0, 64), tl.arange(0, 128)
    rows = group.to(tl.int64) * ROWS + row
    used = row[:, None] < ROWS
    query = tl.load(
        query_ptr + rows[:, None] * query_stride + dim[None, :], mask=used, other=0.0
    )
    keys = tl.load(
        key_ptr
        + block.to(tl.int64) * key_page_stride
        + key[:, None] * key_row_stride
        + dim[None, :]
    )
    scores = tl.dot(query, tl.trans(keys))
    key = page * 64 + key
    tl.store(
        scores_ptr + rows[:, None] * scores_stride + key[None, :],
        scores,
        mask=used & (key[None, :] < length),
    )
