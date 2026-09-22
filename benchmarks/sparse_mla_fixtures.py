# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Native-input adaptation of FlashMLA's sparse decoding test distribution.

Reference: deepseek-ai/FlashMLA ba89a3466e9470ad08ab39738d4e7bb66989e1e7,
tests/lib.py:_randperm_batch and generate_testcase_for_decode. Selection uses
random scores, not model indexer scores. All preparation is outside timing.
"""

import random

import torch


def flashmla_fixture(
    batch,
    heads,
    queries,
    dtype,
    swa_page,
    compressed_page,
    seed,
    *,
    compressed_topk,
    swa_cache_tokens=16384,
    compressed_cache_tokens=16384,
    variable_cache_lengths=True,
    variable_topk_lengths=False,
    device="cuda",
):
    """Return native BF16/FP8 tensors with independent random physical pools.

    Unlike the packed FlashMLA test, caches are contiguous native tensors and
    Q uses the same native dtype as KV. The distribution, page permutation,
    unique random-score top-k, unused-row poisoning, and sink logits follow
    the reference. RNGs are local so fixture construction cannot affect callers.
    """
    if min(batch, heads, queries, swa_cache_tokens, compressed_cache_tokens) <= 0:
        raise ValueError("fixture dimensions and cache lengths must be positive")
    if compressed_topk < 0 or compressed_topk % 4:
        raise ValueError("compressed top-k must be nonnegative and divisible by four")
    generator = torch.Generator(device=device).manual_seed(seed)
    host_generator = random.Random(seed)

    def randn(shape):
        return torch.randn(shape, device=device, generator=generator)

    query = randn((batch, queries, heads, 512)).clamp_(-1, 1).to(dtype)
    sinks = randn((heads,))
    sink_mask = randn((heads,))
    sinks[sink_mask > 0.5] = float("inf")
    sinks[sink_mask < -0.5] = -float("inf")

    def source(mean_length, page, topk, variable_topk, is_swa=False):
        lengths = [
            int(
                max(host_generator.normalvariate(mean_length, mean_length / 2), queries)
            )
            if variable_cache_lengths
            else mean_length
            for _ in range(batch)
        ]
        alignment = 4 * page
        rows = max(1, (max(lengths) + alignment - 1) // alignment) * alignment
        table = torch.randperm(batch * rows // page, device=device, generator=generator)
        table = table.to(torch.int32).view(batch, rows // page)
        cache = (randn((batch * rows // page, page, 512)) / 10).clamp_(-1, 1)
        lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
        # TRT's SWA window has an implicit per-query causal length. Match it
        # in the explicit lists, including requests shorter than 128 tokens.
        ranges = (
            (
                lengths[:, None] - queries + torch.arange(queries, device=device) + 1
            ).reshape(-1)
            if is_swa
            else lengths.repeat_interleave(queries)
        )
        score_columns = max(rows, topk)
        scores = torch.rand(
            (batch * queries, score_columns), device=device, generator=generator
        )
        scores.masked_fill_(
            torch.arange(score_columns, device=device)[None, :] >= ranges[:, None],
            -float("inf"),
        )
        logical = scores.topk(topk, dim=-1, sorted=True).indices.view(
            batch, queries, topk
        )
        valid = logical < ranges.view(batch, queries, 1)
        safe = logical.masked_fill(~valid, 0)
        physical = table.gather(1, (safe // page).reshape(batch, -1)).view_as(safe)
        indices = (physical * page + safe % page).to(torch.int32)
        indices.masked_fill_(~valid, -1)
        selected_lengths = (
            torch.randint(
                0,
                topk + 1,
                (batch, 1),
                device=device,
                dtype=torch.int32,
                generator=generator,
            )
            .expand(batch, queries)
            .contiguous()
            if variable_topk
            else torch.full((batch, queries), topk, device=device, dtype=torch.int32)
        )
        # TRT-LLM consumes the active prefix length; trim trailing -1 slots
        # before giving the identical metadata to either backend. This preserves
        # the selected set and avoids counting padding as attention mass.
        if not is_swa:
            selected_lengths = torch.minimum(
                selected_lengths, valid.sum(-1).to(torch.int32)
            )
        used = valid & (
            torch.arange(topk, device=device)[None, None, :]
            < selected_lengths[..., None]
        )
        # Canonical inactive slots must be OOB, including variable prefixes:
        # some comparators gather V before masking scores, and 0 * NaN is NaN.
        # Active selections and poisoned physical cache values are unchanged.
        indices.masked_fill_(~used, -1)
        unused = torch.ones(batch * rows, device=device, dtype=torch.bool)
        unused[indices[used].long()] = False
        cache.view(-1, 512)[unused] = float("nan")
        return cache.to(dtype), indices, selected_lengths, lengths

    swa, si, sl, seq_lens = source(swa_cache_tokens, swa_page, 128, False, is_swa=True)
    compressed, ci, cl, _ = source(
        compressed_cache_tokens, compressed_page, compressed_topk, variable_topk_lengths
    )
    return dict(
        query=query,
        swa=swa,
        compressed=compressed,
        si=si,
        ci=ci,
        sl=sl,
        cl=cl,
        sinks=sinks,
        seq_lens=seq_lens,
        combined=torch.cat((si, ci), dim=-1).reshape(batch * queries, -1),
        combined_lengths=(sl + cl).reshape(-1),
        softmax_scale=512**-0.55,
    )
