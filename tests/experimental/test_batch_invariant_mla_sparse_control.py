# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Negative control of the batch-invariant sparse MLA tests: today's default ``prims_ts`` schedule fails the
comparison the classes pass. It runs the experimental wrapper, so it lives in the experimental lane; the classes do
not depend on that schedule, and if it stops differing this control has nothing to show."""

import warnings

import torch

from tests.batch_invariant.test_mla_sparse import (
    SCALE,
    TOPK,
    join,
    mates,
    needs_device,
    run_batch,
    same_bits,
    shared,
)
from flashinfer.batch_invariant.mla import cases


def default_wrapper(cfg, rows, width=TOPK):
    """The default (not batch-invariant) ``prims_ts`` schedule, for the negative control."""
    from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

    dtype, head_dim, heads = cfg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrapper = BatchSparseMLADecodePagedTSWrapper()
        wrapper.plan(
            torch.device("cuda"),
            rows,
            heads,
            max_topk=width,
            q_dtype=cases.torch_dtype(dtype),
            qk_rope_head_dim=head_dim - 512,
            softmax_scale=SCALE,
            return_lse=True,
        )
    return wrapper


@needs_device
def test_default_schedule_is_not_batch_invariant():
    """Negative control: on today's default ``prims_ts`` schedule the same comparison fails, so the invariance tests
    can fail. The classes do not depend on that schedule: if it stops differing, this control has nothing to show."""
    cfg = ("e4m3", 576, 64)
    _, head_dim, heads = cfg
    pool, targets, _ = shared(cfg)
    full = targets.take([10, 11])

    def default(rows):
        n = len(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, _ = default_wrapper(cfg, n).run(
                rows.q.view(n, 1, heads, head_dim), pool.kv, rows.indices, rows.lengths
            )
        return out.view(n, heads, 512)

    alone = [default(full.take([i])) for i in range(2)]
    batch = join(full, mates(cfg, pool, 62, "long"))
    assert not same_bits(default(batch)[:2], torch.cat(alone)), (
        "the default prims_ts schedule gave batch-independent bits: the control needs another shape"
    )
    flagged = run_batch(batch)[0][:2]
    assert same_bits(
        flagged, torch.cat([run_batch(full.take([i]))[0] for i in range(2)])
    )
