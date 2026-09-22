# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Fixed-context, causal two-source fixtures and a chunked FP64 oracle.

Top-k follows FlashMLA tests/lib.py:_randperm_batch: rank independent uniform
scores, mask unavailable logical positions, and map through random pages.
All preparation and checking are outside the measured attention invocation.
"""

import hashlib
import math

import torch


def model_fixture(
    batch,
    heads,
    queries,
    topk,
    dtype,
    *,
    raw_tokens=32768,
    compression_ratio=1,
    causal=True,
    seed=2026,
    device="cuda",
    score_chunk=128,
    debug_layout=False,
):
    if min(batch, heads, queries, topk, compression_ratio, score_chunk) <= 0:
        raise ValueError("dimensions and compression ratio must be positive")
    if raw_tokens < queries + 127:
        raise ValueError("the query suffix must have a full preceding SWA window")
    candidate_rows = raw_tokens // compression_ratio
    if candidate_rows == 0:
        raise ValueError("compression ratio leaves no candidate entries")
    g = torch.Generator(device=device).manual_seed(seed)

    def normal_native(shape, scale=1.0):
        result = torch.empty(shape, dtype=dtype, device=device)
        flat = result.view(-1, 512)
        # Avoid a full FP32 copy of a multi-GB native cache.
        for start in range(0, flat.shape[0], 32768):
            dst = flat[start : start + 32768]
            value = torch.randn(dst.shape, generator=g, device=device)
            value.mul_(scale).clamp_(-1, 1)
            dst.copy_(value)
        return result

    query = normal_native((batch, queries, heads, 512))
    sinks = torch.randn(heads, generator=g, device=device)
    sink_mask = torch.randn(heads, generator=g, device=device)
    sinks[sink_mask > 0.5] = torch.inf
    sinks[sink_mask < -0.5] = -torch.inf

    swa_page = 256
    swa_rows = math.ceil((queries + 127) / swa_page) * swa_page
    swa_table = (
        torch.randperm(batch * swa_rows // swa_page, generator=g, device=device)
        .to(torch.int32)
        .view(batch, -1)
    )
    swa = normal_native((batch * swa_rows // swa_page, swa_page, 512), 0.1)
    local_swa = (
        torch.arange(queries, device=device, dtype=torch.int32)[:, None]
        + torch.arange(128, device=device, dtype=torch.int32)[None, :]
    )
    si = (
        swa_table[:, local_swa // swa_page] * swa_page
        + local_swa[None, :, :] % swa_page
    ).contiguous()
    sl = torch.full((batch, queries), 128, dtype=torch.int32, device=device)

    primary_table = (
        torch.randperm(batch * candidate_rows, generator=g, device=device)
        .to(torch.int32)
        .view(batch, candidate_rows)
    )
    compressed = normal_native((batch * candidate_rows, 1, 512), 0.1)
    query_lengths = (
        raw_tokens
        - queries
        + torch.arange(queries, device=device, dtype=torch.int32)
        + 1
    )
    available = (
        query_lengths // compression_ratio
        if causal
        else torch.full_like(query_lengths, candidate_rows)
    )
    ci = torch.empty((batch, queries, topk), dtype=torch.int32, device=device)
    columns = max(candidate_rows, topk)
    positions = torch.arange(columns, device=device)
    for request in range(batch):
        for start in range(0, queries, score_chunk):
            end = min(queries, start + score_chunk)
            scores = torch.rand((end - start, columns), generator=g, device=device)
            scores.masked_fill_(
                positions[None, :] >= available[start:end, None], -torch.inf
            )
            logical = scores.topk(topk, dim=-1, sorted=True).indices
            valid = logical < available[start:end, None]
            physical = primary_table[request, logical.masked_fill(~valid, 0)]
            ci[request, start:end] = physical.masked_fill(~valid, -1)
    cl = available.clamp(max=topk)[None, :].expand(batch, queries).contiguous()

    # Poison unused physical rows, exactly accounting for all query selections.
    for cache, indices in ((swa, si), (compressed, ci)):
        flat = cache.view(-1, 512)
        unused = torch.ones(flat.shape[0], dtype=torch.bool, device=device)
        used = indices.reshape(-1)
        unused[used[used >= 0].long()] = False
        if dtype == torch.float8_e4m3fn:
            # Torch 2.8 lacks masked_fill for native FP8. E4M3FN 0x7f is
            # NaN; write the same native values through their byte view.
            flat.view(torch.uint8)[unused] = 0x7F
        else:
            flat[unused] = torch.nan

    fixture = dict(
        query=query,
        swa=swa,
        compressed=compressed,
        si=si,
        ci=ci,
        sl=sl,
        cl=cl,
        sinks=sinks,
        seq_lens=torch.full((batch,), raw_tokens, dtype=torch.int32, device=device),
        combined=torch.cat((si, ci), dim=-1).reshape(batch * queries, -1),
        combined_lengths=(sl + cl).reshape(-1),
        softmax_scale=512**-0.55,
    )
    info = dict(
        raw_kv_tokens=raw_tokens,
        compression_ratio=compression_ratio,
        candidate_rows_per_request=candidate_rows,
        swa_resident_rows_per_request=swa_rows,
        query_start_position=raw_tokens - queries,
        causal=causal,
        valid_topk_min=int(cl.min().item()),
        valid_topk_max=int(cl.max().item()),
        score_chunk=score_chunk,
    )
    if debug_layout:
        info.update(swa_table=swa_table, primary_table=primary_table)
    return fixture, info


def chunked_reference(fixture, *, chunk_rows=32):
    """Return FP64 O/LSE and the same error budget as sparse_mla_reference.

    Batching independent rows bounds selected-KV temporary storage while
    removing thousands of Python/GPU synchronizations for prefill validation.
    No reference work is captured in the benchmark graphs.
    """
    q = fixture.query
    rows, heads = math.prod(q.shape[:-2]), q.shape[-2]
    out = torch.empty((rows, heads, 512), dtype=torch.float64, device=q.device)
    lse_out = torch.empty((rows, heads), dtype=torch.float64, device=q.device)
    bound = torch.empty_like(out)
    sources = (
        (fixture.swa, fixture.si.reshape(rows, -1), fixture.sl.reshape(-1)),
        (fixture.compressed, fixture.ci.reshape(rows, -1), fixture.cl.reshape(-1)),
    )
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        values, valid_parts = [], []
        for cache, indices, lengths in sources:
            idx = indices[start:end]
            valid = (idx >= 0) & (
                torch.arange(idx.shape[-1], device=q.device)[None, :]
                < lengths[start:end, None]
            )
            safe = idx.masked_fill(~valid, 0).long()
            kv = cache.view(-1, 512)[safe].double()
            kv.masked_fill_(~valid[..., None], 0)
            values.append(kv)
            valid_parts.append(valid)
        kv = torch.cat(values, dim=1)
        valid = torch.cat(valid_parts, dim=1)
        empty = ~valid.any(-1)
        qq = q.reshape(rows, heads, 512)[start:end].double()
        logits = torch.bmm(qq, kv.transpose(1, 2)) * fixture.softmax_scale
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        lse = logits.logsumexp(-1)
        normalizer = torch.logaddexp(lse, fixture.sinks.double()[None, :])
        normalizer.masked_fill_(empty[:, None], 0)
        weights = (logits - normalizer[..., None]).exp()
        expected = torch.bmm(weights, kv)
        out[start:end] = expected
        lse_out[start:end] = lse
        if q.dtype == torch.float8_e4m3fn:
            absolute_v = kv.abs()
            sensitivity = torch.bmm(weights, absolute_v)
            underflow = (logits.amax(-1) - normalizer).exp()[
                ..., None
            ] * absolute_v.sum(1)[:, None, :]
            budget = (
                (2**-4 + 3 * 2**-8 + 1e-5) * sensitivity
                + (2**-10 / 448) * underflow
                + 1e-6
            )
            budget.masked_fill_(empty[:, None, None], 0)
        else:
            budget = expected.abs() * 0.02 + 8e-4
        bound[start:end] = budget
    return (
        out.reshape(*q.shape[:-1], 512),
        lse_out.reshape(q.shape[:-1]),
        bound.reshape(*q.shape[:-1], 512),
    )


def streaming_fingerprint(fixture, *, chunk_bytes=64 * 1024 * 1024):
    """Hash the same bytes/layout as the original helper, with bounded host RAM."""
    digest = hashlib.sha256()
    for name, value in sorted(vars(fixture).items()):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            digest.update(
                str((value.dtype, tuple(value.shape), value.stride())).encode()
            )
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
            for start in range(0, raw.numel(), chunk_bytes):
                data = raw[start : start + chunk_bytes].cpu().numpy()
                digest.update(memoryview(data))
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()
