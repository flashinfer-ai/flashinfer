# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Prepare the paged/raw-index fixtures used by sparse MLA kernel tests."""

from weakref import WeakKeyDictionary

import torch
import pytest

pytest.importorskip("cutlass", minversion="4.7.0")
pytest.importorskip("triton")
from flashinfer.testing.sparse_mla_metadata import prepare_sparse_mla_metadata

_buffers = WeakKeyDictionary()


def prepare_fixture(
    wrapper,
    query,
    kv_cache,
    extra_kv_cache=None,
    *,
    swa_indices,
    compressed_indices=None,
    swa_topk_lens=None,
    compressed_topk_lens=None,
    swa_kv_scale=1.0,
    compressed_kv_scale=None,
    **kwargs,
):
    """Refresh external metadata and return arguments for the public run().

    Cache only allocation geometry, so warmup and capture use stable addresses.
    Fixture mutations are mapped by Triton during every graph replay.
    """
    extra_scale = swa_kv_scale if compressed_kv_scale is None else compressed_kv_scale
    shared = extra_scale is swa_kv_scale or (
        not isinstance(extra_scale, torch.Tensor)
        and not isinstance(swa_kv_scale, torch.Tensor)
        and extra_scale == swa_kv_scale
    )
    state = wrapper._impl._state
    previous, cached = _buffers.get(wrapper, (None, {}))
    if previous is not state:
        cached = {}
        _buffers[wrapper] = (state, cached)
    key = (tuple(query.shape), shared)
    common = {
        k: kwargs[k]
        for k in ("softmax_scale", "q_scale", "output_scale", "sinks")
        if k in kwargs
    }
    metadata = prepare_sparse_mla_metadata(
        wrapper,
        query,
        kv_cache,
        swa_indices,
        swa_topk_lens,
        extra_kv_cache=extra_kv_cache,
        extra_indices=compressed_indices,
        extra_lengths=compressed_topk_lens,
        kv_scale=swa_kv_scale,
        extra_kv_scale=extra_scale,
        out=cached.get(key),
        **common,
    )
    cached[key] = metadata
    return dict(
        query=query,
        kv_cache=kv_cache,
        extra_kv_cache=extra_kv_cache,
        metadata=metadata,
        kv_scale=swa_kv_scale,
        extra_kv_scale=extra_scale,
        **kwargs,
    )
