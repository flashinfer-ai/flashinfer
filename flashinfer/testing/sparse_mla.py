# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent, deliberately unoptimized native two-source MLA reference.

Indices are physical ``page * page_size + offset`` values, shared by the heads
of one query token. The reference never infers causality from physical indices.
These helpers are for validation/fixture preparation, outside timed regions.
"""

import math

import torch


def sparse_mla_reference(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor | None,
    swa_indices: torch.Tensor,
    compressed_indices: torch.Tensor | None = None,
    *,
    swa_topk_lens: torch.Tensor | None = None,
    compressed_topk_lens: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    q_scale: float = 1.0,
    swa_kv_scale: float = 1.0,
    compressed_kv_scale: float = 1.0,
    output_scale: float = 1.0,
    sinks: torch.Tensor | None = None,
    reference_dtype: torch.dtype = torch.float64,
    return_fp8_error_bound: bool = False,
    chunk_rows: int = 32,
) -> tuple[torch.Tensor, ...]:
    """Return unrounded O and natural-log LSE, excluding sink from LSE.

    Q may be fixed ``[B,Sq,H,D]`` or packed ``[total_q,H,D]``. Cache tensors
    have shape ``[pages,page_size,D]``; their page stride may contain padding.
    Lengths are per query token, and ``-1`` entries inside a prefix are masked.
    An empty attention row has O=0 and LSE=-inf, even with an infinite sink.
    Accumulation is FP64 by default, so the oracle does not use tensor cores.
    With return_fp8_error_bound, also return a componentwise forward-error
    budget for E4M3 probability rounding plus up to three BF16 output/partial
    roundings. It scales with attention-weighted absolute V, so cancellation
    does not incorrectly imply that arithmetic error must also cancel.
    """
    if reference_dtype not in (torch.float32, torch.float64):
        raise ValueError("reference_dtype must be float32 or float64")
    if query.ndim not in (3, 4):
        raise ValueError("query must have rank 3 or 4")
    if query.shape[-1] not in (512, 576):
        raise ValueError("query dimension must be 512 or 576")
    scales = (q_scale, swa_kv_scale, compressed_kv_scale, output_scale)
    if not all(math.isfinite(x) and x > 0 for x in scales):
        raise ValueError("descales and output_scale must be positive and finite")
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax_scale must be positive and finite")
    if (compressed_kv_cache is None) != (compressed_indices is None):
        raise ValueError("compressed cache and indices must be provided together")
    if compressed_kv_cache is None and compressed_topk_lens is not None:
        raise ValueError("compressed lengths require a compressed cache")
    rows = math.prod(query.shape[:-2])
    heads, dim = query.shape[-2:]
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    q = query.reshape(rows, heads, dim)
    out = torch.zeros((rows, heads, 512), device=query.device, dtype=reference_dtype)
    lse = torch.full(
        (rows, heads), -torch.inf, device=query.device, dtype=reference_dtype
    )
    error_bound = torch.zeros_like(out) if return_fp8_error_bound else None
    if sinks is not None:
        if sinks.shape != (heads,) or torch.isnan(sinks).any():
            raise ValueError("sinks must have one non-NaN logit per head")
        sinks = sinks.to(reference_dtype)

    sources = []
    for cache, indices, lengths, descale in (
        (swa_kv_cache, swa_indices, swa_topk_lens, swa_kv_scale),
        (
            compressed_kv_cache,
            compressed_indices,
            compressed_topk_lens,
            compressed_kv_scale,
        ),
    ):
        if cache is None:
            continue
        if cache.ndim != 3 or cache.shape[-1] != dim or cache.shape[1] <= 0:
            raise ValueError("cache must have shape [pages,page_size,Dqk]")
        if indices.dtype != torch.int32 or tuple(indices.shape[:-1]) != tuple(
            query.shape[:-2]
        ):
            raise ValueError("indices must be int32 with one list per query token")
        capacity = indices.shape[-1]
        idx = indices.reshape(rows, capacity)
        if lengths is None:
            lens = torch.full((rows,), capacity, device=query.device, dtype=torch.int32)
        else:
            if lengths.dtype != torch.int32 or tuple(lengths.shape) != tuple(
                query.shape[:-2]
            ):
                raise ValueError("lengths must be int32 with one value per query token")
            lens = lengths.reshape(rows)
            if ((lens < 0) | (lens > capacity)).any():
                raise ValueError("source length exceeds index capacity")
        sources.append((cache, idx, lens, descale))

    # Batch independent query rows while bounding selected-KV temporary storage.
    # Invalid slots are zeroed before matmul, including poisoned page padding.
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        values, masks = [], []
        for cache, indices, lengths, descale in sources:
            idx = indices[start:end].to(torch.int64)
            active = (
                torch.arange(idx.shape[-1], device=query.device)[None, :]
                < lengths[start:end, None]
            )
            if (active & ((idx < -1) | (idx >= cache.shape[0] * cache.shape[1]))).any():
                raise ValueError("active index is outside its source pool")
            valid = active & (idx >= 0)
            safe = idx.masked_fill(~valid, 0)
            selected = (
                cache[safe // cache.shape[1], safe % cache.shape[1]].to(reference_dtype)
                * descale
            )
            selected.masked_fill_(~valid[..., None], 0)
            values.append(selected)
            masks.append(valid)
        kv, valid = torch.cat(values, dim=1), torch.cat(masks, dim=1)
        empty = ~valid.any(-1)
        logits = (
            torch.bmm(q[start:end].to(reference_dtype) * q_scale, kv.transpose(1, 2))
            * softmax_scale
        )
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        row_lse = logits.logsumexp(-1)
        normalizer = (
            row_lse if sinks is None else torch.logaddexp(row_lse, sinks[None, :])
        )
        normalizer = normalizer.masked_fill(empty[:, None], 0)
        weights = (logits - normalizer[..., None]).exp()
        out[start:end] = torch.bmm(weights, kv[..., :512]) * output_scale
        lse[start:end] = row_lse
        if error_bound is not None:
            # RN E4M3: relative error <= 2^-4, minimum subnormal 2^-9.
            # P uses scale 448; include three BF16 partial/output roundings.
            absolute_v = kv[..., :512].abs()
            sensitivity = torch.bmm(weights, absolute_v)
            underflow = (logits.amax(-1) - normalizer).exp()[
                ..., None
            ] * absolute_v.sum(1)[:, None, :]
            error_bound[start:end] = (
                output_scale
                * (
                    (2**-4 + 3 * 2**-8 + 1e-5) * sensitivity
                    + (2**-10 / 448) * underflow
                )
                + 1e-6
            ).masked_fill(empty[:, None, None], 0)
    result = (out.reshape(*query.shape[:-1], 512), lse.reshape(*query.shape[:-1]))
    if error_bound is not None:
        return (*result, error_bound.reshape(*query.shape[:-1], 512))
    return result
