# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Check each selection against FP32 input scores, without an index-order contract."""

import torch


def assert_topk_reference(
    logits,
    indices,
    seq_lens,
    top_k,
    *,
    next_n=1,
    compress_ratio=1,
    row_starts=None,
    absolute_indices=False,
    page_table=None,
    page_size=1,
    values=None,
):
    """Validate every row, including short/empty rows and boundary ties.

    Input logits are already the scores being ranked, so their FP32 conversion
    is an exact reference. Check selected score multisets, not another run or
    one arbitrary tie-breaking index set. Paged output is inverted only through
    live pages. Returned values must correspond to their own returned indices.
    """
    rows, width = logits.shape
    assert indices.shape == (rows, top_k)
    assert indices.dtype in (torch.int32, torch.int64)
    lengths = seq_lens.cpu().tolist()
    starts = None if row_starts is None else row_starts.cpu().tolist()
    tables = None if page_table is None else page_table.cpu().tolist()
    scores = logits.float()
    for row in range(rows):
        request = row // next_n
        if starts is None:
            start = 0
            length = (lengths[request] - next_n + row % next_n + 1) // compress_ratio
            length = max(0, min(width, length))
        else:
            start = max(0, min(width, starts[row]))
            length = max(0, min(width, start + lengths[row]) - start)
        count = min(top_k, length)
        result = indices[row].long()
        assert bool((result[count:] == -1).all()), f"row {row}: invalid padding"
        head = result[:count]
        assert bool((head >= 0).all()), f"row {row}: negative selected index"
        assert head.unique().numel() == count, f"row {row}: duplicate selected index"
        if tables is not None:
            size = 1 if page_size is None else page_size
            live_pages = tables[request][: (length + size - 1) // size]
            inverse = {physical: logical for logical, physical in enumerate(live_pages)}
            assert len(inverse) == len(live_pages), "test page map must be injective"
            logical = []
            for slot in head.cpu().tolist():
                physical, offset = divmod(slot, size)
                assert physical in inverse, f"row {row}: slot outside live page map"
                logical.append(inverse[physical] * size + offset)
            head = torch.tensor(logical, dtype=torch.int64, device=logits.device)
        elif absolute_indices:
            head = head - start
        assert bool((head < length).all()), f"row {row}: selected index past window"
        assert bool((head >= 0).all()), f"row {row}: selected index before window"
        window = scores[row, start : start + length]
        selected = window[head]
        expected = window.topk(count).values
        torch.testing.assert_close(
            selected.sort().values,
            expected.sort().values,
            rtol=0,
            atol=0,
            msg=lambda message: f"row {row}: {message}",
        )
        if values is not None:
            torch.testing.assert_close(
                values[row, :count].float(), selected, rtol=0, atol=0
            )
