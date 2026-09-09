"""plan() derives _max_q_len and _max_kv_len with a tensor reduction.

These two attributes used to be computed with the Python builtin max() over a
1-D int32 host tensor, which iterates element by element. The reduction must
produce exactly the same Python int, so pin both against the builtin over a
range of batch sizes, page sizes and dtypes, including the single-request
case where the reduced tensor has one element.
"""

import pytest
import torch

from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.page import get_seq_lens


@pytest.mark.parametrize("batch_size", [1, 3, 8, 17])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_plan_max_lens(batch_size, page_size, dtype):
    torch.manual_seed(batch_size * 131 + page_size)
    q_lens = torch.randint(1, 65, (batch_size,), dtype=torch.int32)
    kv_lens = torch.maximum(
        torch.randint(1, 513, (batch_size,), dtype=torch.int32), q_lens
    )

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(q_lens, 0)
    num_pages = (kv_lens + page_size - 1) // page_size
    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    paged_kv_indptr[1:] = torch.cumsum(num_pages, 0)
    paged_kv_indices = torch.arange(int(paged_kv_indptr[-1]), dtype=torch.int32)
    paged_kv_last_page_len = kv_lens - (num_pages - 1) * page_size

    # What the code computed before the rewrite, spelled the old way.
    expected_q = max(qo_indptr[1:] - qo_indptr[:-1]).item()
    expected_kv = max(
        get_seq_lens(paged_kv_indptr, paged_kv_last_page_len, page_size)
    ).item()

    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)
    wrapper.plan(
        qo_indptr.cuda(),
        paged_kv_indptr.cuda(),
        paged_kv_indices.cuda(),
        paged_kv_last_page_len.cuda(),
        8,
        8,
        128,
        page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    assert isinstance(wrapper._max_q_len, int)
    assert wrapper._max_q_len == expected_q
    assert isinstance(wrapper._max_kv_len, int)
    assert wrapper._max_kv_len == expected_kv
    # And they are what the inputs say they are, independent of either spelling.
    assert wrapper._max_q_len == int(q_lens.max())
    assert wrapper._max_kv_len == int(kv_lens.max())
