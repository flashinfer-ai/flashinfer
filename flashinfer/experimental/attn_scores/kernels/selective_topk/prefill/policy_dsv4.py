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

"""Shape-based C4/H64 sampling, using the shared Qwen stratified method."""

import cutlass
import cutlass.cute as cute
from .policy_glm_short import (
    runtime_samples as previous_samples,
    model_proposal_rank as previous_rank,
)
from .policy_q4_128 import individual_sample_key as previous_key
from .policy_qwen import runtime_samples as qwen_samples
from .sample_policy import proposal_rank


@cute.jit
def runtime_samples(
    length,
    k: cutlass.Constexpr = 2048,
    heads: cutlass.Constexpr = 64,
    coverage: cutlass.Constexpr = 8,
    dsv4: cutlass.Constexpr = False,
    fp4: cutlass.Constexpr = False,
):
    if cutlass.const_expr(dsv4):
        samples = qwen_samples(length, k, heads=4, coverage=4)
        if cutlass.const_expr(fp4):
            # MXFP4 sampler: the halved count is kept only while it still expects >= 2.7 top-k hits
            # (10*k*half >= 27*length). Below that, the rank margin mu+4.5*sqrt(mu)+1 widens the threshold
            # (count/k ~ rank/mu: 4.7 at mu=2 vs 3.5 at mu=4) and the extra K3/K2 candidates cost more than the K1
            # samples saved (B200, K buffer = 1M tokens, K1+K2+K3: DS-V4-Pro 196K/256K -29/-53 us, 393-524K -9..-10 us; DS-V4-Flash
            # 96-128K -8..-16 us; same TopK). DS-V4-Flash keeps the halved count when the full one exceeds 256 samples
            # (+256 samples cost 29 us of K1 against 12-17 us of K3). The FP8 sampler keeps the halved counts.
            if cutlass.const_expr(k == 512):
                if length <= 128 * k:
                    half = max(cutlass.Int32(128), samples // 2)
                    if (samples > 256) | (
                        cutlass.Int32(10 * k) * half >= cutlass.Int32(27) * length
                    ):
                        samples = half
            else:
                if (length >= 32 * k) & (length <= 128 * k):
                    if (
                        cutlass.Int32(10 * k) * (samples // 2)
                        >= cutlass.Int32(27) * length
                    ):
                        samples = samples // 2
        elif cutlass.const_expr(k == 512):
            if length <= 128 * k:
                samples = max(cutlass.Int32(128), samples // 2)
        else:
            if (length >= 32 * k) & (length <= 128 * k):
                samples = samples // 2
        return samples
    else:
        return previous_samples(length, k, heads, coverage)


@cute.jit
def block_samples(
    qb,
    ends,
    nq,
    bq: cutlass.Constexpr,
    k: cutlass.Constexpr = 2048,
    heads: cutlass.Constexpr = 64,
    coverage: cutlass.Constexpr = 8,
    dsv4: cutlass.Constexpr = False,
    anchor_upper=0,
    fp4: cutlass.Constexpr = False,
):
    length = ends[min(nq - 1, qb * bq + bq - 1)]
    if cutlass.const_expr(dsv4 and k == 1024):
        if anchor_upper > 0:
            length = anchor_upper
    return runtime_samples(length, k, heads, coverage, dsv4, fp4)


@cute.jit
def individual_sample_key(
    pos,
    qb,
    starts,
    ends,
    nq,
    bq: cutlass.Constexpr,
    k: cutlass.Constexpr = 2048,
    dsv4: cutlass.Constexpr = False,
    fp4: cutlass.Constexpr = False,
):
    if cutlass.const_expr(dsv4):
        first = qb * bq
        lower = starts[min(first, nq - 1)]
        for token in cutlass.range_constexpr(1, bq):
            lower = min(lower, starts[min(nq - 1, first + token)])
        upper = ends[min(nq - 1, first + bq - 1)]
        return dsv4_cached_sample_key(pos, qb, lower, upper, k, coverage=4, fp4=fp4)
    else:
        return previous_key(pos, qb, starts, ends, nq, bq, k)


@cute.jit
def model_proposal_rank(
    width,
    samples,
    k: cutlass.Constexpr,
    sigma: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    dsv4: cutlass.Constexpr = False,
):
    if cutlass.const_expr(dsv4):
        rank = proposal_rank(width, samples, k, sigma)
        if samples <= 256:
            rank = proposal_rank(width, samples, k, 4.5)
        return rank
    else:
        return previous_rank(width, samples, k, sigma, heads)


from .policy_qwen import sample_hash_u32


@cute.jit
def dsv4_cached_sample_key(
    pos,
    qb,
    lower,
    upper,
    k: cutlass.Constexpr = 512,
    coverage: cutlass.Constexpr = 4,
    sample_count: cutlass.Constexpr = 0,
    fp4: cutlass.Constexpr = False,
):
    width = max(1, upper - lower)
    samples = (
        cutlass.Int32(sample_count)
        if cutlass.const_expr(sample_count > 0)
        else runtime_samples(upper, k, dsv4=True, fp4=fp4)
    )
    shift = 31 - cutlass.Int32(cute.arch.clz(cutlass.Uint32(samples)))
    h = sample_hash_u32(pos, qb)
    begin = cutlass.Int32((cutlass.Int64(pos) * cutlass.Int64(width)) >> shift)
    finish = cutlass.Int32((cutlass.Int64(pos + 1) * cutlass.Int64(width)) >> shift)
    span = max(1, finish - begin)
    offset = begin + cutlass.Int32((cutlass.Uint64(h) * cutlass.Uint64(span)) >> 32)
    return max(cutlass.Int32(0), min(upper - 1, lower + offset))


from .policy_glm_short import model_sample_key as model_sample_key
