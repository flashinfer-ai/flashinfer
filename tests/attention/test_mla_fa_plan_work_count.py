"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random

import pytest

import flashinfer.mla._batch_mla._backends._fa_common as fa_common


def _reference_work_count(q_lens, kv_lens, num_heads, num_sm, causal):
    """Per-request mirror of scheduler.cuh MLAPlan, without any shortcuts."""
    cluster = 2 if sum(q_lens) * num_heads // len(q_lens) > 64 else 1
    clusters = num_sm // cluster
    tile = 64 * cluster
    effective = []
    for q, kv in zip(q_lens, kv_lens, strict=True):
        count = (q * num_heads + tile - 1) // tile
        for index in range(count):
            if causal and index + 1 != count:
                visible = kv - q + ((index + 1) * tile + num_heads - 1) // num_heads
                effective.append(max(min(visible, kv), 0))
            else:
                effective.append(kv)
    average = max((sum(effective) + clusters - 1) // clusters, 1)
    if average <= 8:
        limit = 32
    elif average <= 16:
        limit = 64
    elif average <= 32:
        limit = 128
    elif average <= 64:
        limit = 192
    else:
        limit = (average + 255) // 256 * 256
    return sum(max(1, (kv + limit - 1) // limit) for kv in effective)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("live_fraction", [1.0, 0.25, 0.01])
def test_fa_plan_work_count_matches_reference(causal, live_fraction):
    """Zero-query padding rows (as CUDA-graph callers plan them) must not
    change the count, beyond their effect on the cluster-size choice. Shapes
    stay within the native work-item capacity."""
    rng = random.Random(0)
    for _ in range(200):
        rows = rng.randint(1, 512 if live_fraction == 1.0 else 8192)
        num_heads = rng.choice([1, 8, 16, 64, 128])
        q_lens = [
            rng.randint(1, 4) if rng.random() < live_fraction else 0
            for _ in range(rows)
        ]
        kv_lens = [q + rng.randint(0, 65536) if q else 0 for q in q_lens]
        num_sm = rng.choice([2, 78, 132])
        expected = _reference_work_count(q_lens, kv_lens, num_heads, num_sm, causal)
        assert (
            fa_common._fa_plan_work_count(q_lens, kv_lens, num_heads, num_sm, causal)
            == expected
        )


def test_fa_plan_work_count_cuda_graph_padded_decode():
    """A decode batch planned into 8192-row CUDA-graph buffers."""
    q_lens = [1] * 64 + [0] * (8192 - 64)
    kv_lens = [2048] * 64 + [0] * (8192 - 64)
    assert fa_common._fa_plan_work_count(
        q_lens, kv_lens, 64, 132, False
    ) == _reference_work_count(q_lens, kv_lens, 64, 132, False)
