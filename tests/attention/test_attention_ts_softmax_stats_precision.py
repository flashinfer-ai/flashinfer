# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Large finite logits must not erase the merge-compatible denominator."""

import math

import pytest
import torch


def test_absolute_fp32_lse_cannot_recover_denominator():
    """Keep the adversarial input independent of GPU availability and dispatch."""
    maximum = torch.tensor(4096.0 * 4096.0, dtype=torch.float32)
    max_log2 = maximum * math.log2(math.e)
    denominator = torch.tensor(2.0, dtype=torch.float32)
    lse = max_log2 + denominator.log2()
    assert torch.exp2(lse - max_log2).item() == 4.0
    assert denominator.item() == 2.0


def test_absolute_fp32_scaled_maxima_cannot_preserve_relative_gap():
    raw_maxima = torch.tensor([2**24, 2**24 + 2], dtype=torch.float32)
    scaled_maxima = raw_maxima * math.log2(math.e)
    premature = torch.exp2(scaled_maxima - scaled_maxima.max()).sum() * 256
    correct = torch.exp(raw_maxima - raw_maxima.max()).sum() * 256
    assert premature.item() == 320
    torch.testing.assert_close(correct, torch.tensor(256 * (1 + math.exp(-2))))


def test_absolute_fp32_fma_cannot_preserve_probability_origin():
    raw_maximum = torch.tensor(2**24 + 2, dtype=torch.float32)
    scale_log2 = torch.tensor(math.log2(math.e), dtype=torch.float32)
    # FP64 models the product retained by FMA before its final FP32 rounding.
    residual = (
        raw_maximum.double() * scale_log2.double() - (raw_maximum * scale_log2).double()
    ).float()
    assert torch.exp2(residual).item() > 1.84
    assert torch.exp2((raw_maximum - raw_maximum) * scale_log2).item() == 1


@pytest.mark.parametrize("store_stats", (False, True))
def test_attention_ts_mla_cluster_stats_storage_is_opt_in(store_stats):
    pytest.importorskip("cutlass", minversion="4.7.0")
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
        make_throughput_latency_mla_config,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.resources.tmem_corr import (
        TmemCorrResource,
    )

    cfg = make_throughput_latency_mla_config(
        batch_size=1,
        num_heads_q=8,
        seq_len_q=1,
        seq_len_kv=8193,
        tile_size_q=8,
        explicit_split_kv=4,
        explicit_persistent=False,
        max_active_clusters=148,
        reduction_mode="cluster",
    )
    resource = TmemCorrResource(
        name="corr",
        cfg=cfg,
        inst_id=1,
        softmax_stats=object() if store_stats else None,
    )
    allocations = resource.get_smem_requirements()
    cluster = next(a for a in allocations if a.name.endswith("_clusterReduction"))
    expected = max(
        cfg.cluster_reduction_smem_bytes_for(s, store_softmax_stats=store_stats)
        for s in range(2, cfg.num_ctas_per_seq_kv + 1)
    )
    assert cluster.size_bytes == expected
    assert (expected > cfg.cluster_reduction_smem_bytes) is store_stats


_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS attention requires SM100 or SM103",
)


def _make_large_logit_case(
    monkeypatch, family, reduction, *, max_kv_len=None, store_stats=True
):
    pytest.importorskip("cutlass", minversion="4.7.0")
    from flashinfer.attention.prims_ts import BatchMLADecodePagedTSWrapper
    from tests.attention.test_attention_ts_softmax_stats import _force_mla_topology

    caches = _force_mla_topology(monkeypatch, family, reduction)
    for cache in caches:
        cache.cache_clear()
    heads = 8 if family == "1cta" else 128
    if max_kv_len is None:
        max_kv_len = 128 if reduction == "direct" else 8193
    page_size = 32
    pages = (max_kv_len + page_size - 1) // page_size
    query = torch.zeros((1, 1, heads, 576), device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros((pages, page_size, 576), device="cuda", dtype=torch.bfloat16)
    # Keep the large dot-product coordinate outside V so output stays small.
    query[..., 512] = 4096
    kv[..., 512] = 4096
    kv[..., 0] = 1
    table = torch.arange(pages, dtype=torch.int32, device="cuda")[None, :]
    lengths = torch.tensor([max_kv_len], dtype=torch.int32, device="cuda")
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        device="cuda",
        batch_size=1,
        num_heads=heads,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        page_size=page_size,
        max_kv_len=max_kv_len,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        mask_type="dense",
        store_softmax_stats=store_stats,
    )
    policy = dict(wrapper._plan_state.policy)
    assert policy["kernel"] == (
        "throughput_latency_1cta" if family == "1cta" else "throughput_2cta"
    )
    assert (policy["split_kv"] == 1) == (reduction == "direct")
    assert policy["use_cluster_reduction"] == (reduction == "cluster")
    return caches, wrapper, query, kv, table, lengths


@pytest.mark.parametrize("family", ("1cta", "2cta"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_stats_probability_origin_and_default_control(
    monkeypatch, family
):
    caches, wrapper, query, kv, table, lengths = _make_large_logit_case(
        monkeypatch, family, "direct"
    )
    try:
        lengths.fill_(2)
        query[..., 513] = 1
        kv[..., 513] = 2
        kv[0, 1, 0] = 4
        stats = torch.empty((*query.shape[:-1], 2), device="cuda")
        for scale in (1.0, 0.5):
            output = wrapper.run(
                query, kv, table, lengths, bmm1_scale=scale, softmax_stats=stats
            )
            expected = torch.zeros_like(output)
            expected[..., 0] = 2.5
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                stats[..., 0],
                torch.full_like(stats[..., 0], (2**24 + 2) * scale),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                stats[..., 1], torch.full_like(stats[..., 1], 2), rtol=2e-5, atol=0
            )

        # Ordinary inputs retain the disabled path's output and match the
        # oracle; this is separate from extreme inputs that expose its old FMA.
        _, default, _, _, _, _ = _make_large_logit_case(
            monkeypatch, family, "direct", store_stats=False
        )
        query[..., 512] = 1
        kv[..., 512] = 1
        kv[0, 0, 513] = 0
        for scale in (1.0, 0.5):
            output = wrapper.run(
                query, kv, table, lengths, bmm1_scale=scale, softmax_stats=stats
            )
            default_output = default.run(query, kv, table, lengths, bmm1_scale=scale)
            expected = torch.zeros_like(output)
            expected[..., 0] = (math.exp(-2 * scale) + 4) / (math.exp(-2 * scale) + 1)
            torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
            torch.testing.assert_close(output, default_output, rtol=0.01, atol=0.01)
            torch.testing.assert_close(
                stats[..., 0],
                torch.full_like(stats[..., 0], 3 * scale),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                stats[..., 1],
                torch.full_like(stats[..., 1], 1 + math.exp(-2 * scale)),
                rtol=2e-5,
                atol=2e-5,
            )
    finally:
        for cache in caches:
            cache.cache_clear()


@pytest.mark.parametrize(
    "family,reduction",
    [("1cta", mode) for mode in ("direct", "cluster", "serial", "parallel")]
    + [("2cta", mode) for mode in ("direct", "serial", "parallel")],
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_stats_large_identical_logits(monkeypatch, family, reduction):
    caches, wrapper, query, kv, table, lengths = _make_large_logit_case(
        monkeypatch, family, reduction
    )
    try:
        stats = torch.full((*query.shape[:-1], 2), torch.nan, device="cuda")
        # Short lengths exercise inactive/pruned split slots on the same plan.
        for token_count in (2, 1) if reduction == "direct" else (8193, 769, 2):
            lengths.fill_(token_count)
            stats.fill_(torch.nan)
            wrapper.run(query, kv, table, lengths, bmm1_scale=1.0, softmax_stats=stats)
            torch.testing.assert_close(
                stats[..., 0],
                torch.full_like(stats[..., 0], 2**24),
                rtol=2e-7,
                atol=0,
            )
            torch.testing.assert_close(
                stats[..., 1],
                torch.full_like(stats[..., 1], token_count),
                rtol=2e-5,
                atol=2e-5,
            )
    finally:
        for cache in caches:
            cache.cache_clear()


@pytest.mark.parametrize("family", ("1cta", "2cta"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_stats_large_logit_partition_merge(monkeypatch, family):
    caches, wrapper, query, kv, table, lengths = _make_large_logit_case(
        monkeypatch, family, "direct"
    )
    try:
        outputs, states = [], []
        query[..., 513] = 1
        for token_count, value, logit_offset in ((1, 1, 0), (2, 4, 2)):
            lengths.fill_(token_count)
            kv[..., 0] = value
            kv[..., 513] = logit_offset
            stats = torch.full((*query.shape[:-1], 2), torch.nan, device="cuda")
            outputs.append(
                wrapper.run(
                    query, kv, table, lengths, bmm1_scale=1.0, softmax_stats=stats
                ).float()
            )
            states.append(stats)
        maximum = torch.maximum(states[0][..., 0], states[1][..., 0])
        weights = [
            state[..., 1] * torch.exp(state[..., 0] - maximum) for state in states
        ]
        merged = sum(
            weight[..., None] * output
            for weight, output in zip(weights, outputs, strict=True)
        )
        merged /= (weights[0] + weights[1])[..., None]
        expected = torch.zeros_like(merged)
        expected[..., 0] = (math.exp(-2) + 8) / (math.exp(-2) + 2)
        torch.testing.assert_close(merged, expected, rtol=2e-5, atol=2e-5)
    finally:
        for cache in caches:
            cache.cache_clear()


@pytest.mark.parametrize(
    "family,reduction",
    [("1cta", mode) for mode in ("cluster", "serial", "parallel")]
    + [("2cta", mode) for mode in ("serial", "parallel")],
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_stats_large_unequal_logits(monkeypatch, family, reduction):
    token_count = 8192
    caches, wrapper, query, kv, table, lengths = _make_large_logit_case(
        monkeypatch, family, reduction, max_kv_len=token_count
    )
    try:
        query[..., 513] = 1
        kv[token_count // (2 * kv.shape[1]) :, :, 513] = 2
        stats = torch.full((*query.shape[:-1], 2), torch.nan, device="cuda")
        # Change the runtime scale without replanning to verify reducer wiring.
        for scale in (1.0, 0.5):
            stats.fill_(torch.nan)
            wrapper.run(
                query, kv, table, lengths, bmm1_scale=scale, softmax_stats=stats
            )
            torch.testing.assert_close(
                stats[..., 0],
                torch.full_like(stats[..., 0], (2**24 + 2) * scale),
                rtol=0,
                atol=0,
            )
            expected_sum = token_count / 2 * (1 + math.exp(-2 * scale))
            torch.testing.assert_close(
                stats[..., 1],
                torch.full_like(stats[..., 1], expected_sum),
                rtol=2e-5,
                atol=2e-5,
            )
    finally:
        for cache in caches:
            cache.cache_clear()
