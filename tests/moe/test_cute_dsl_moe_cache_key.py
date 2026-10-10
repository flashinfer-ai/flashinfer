# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from flashinfer.cute_dsl import is_cute_dsl_available

cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuteDSL not available"
)


def _runner(runner_cls, **parallel):
    kwargs = dict(
        num_experts=256,
        top_k=8,
        num_local_experts=32,
        local_expert_offset=0,
    )
    kwargs.update(parallel)
    if runner_cls.__name__ == "CuteDslFusedMoERunner":
        return runner_cls(forward_impl=lambda *a, **k: None, **kwargs)
    return runner_cls(**kwargs)


@cute_dsl_available
@pytest.mark.parametrize(
    "runner_name",
    ["CuteDslFusedMoERunner", "CuteDslFusedMoEW4A16Runner"],
)
def test_cute_dsl_moe_cache_key_is_rank_invariant(runner_name):
    """Ranks tuned together must share persisted tactics: a cache hit skips the
    set_autotune_process_group reduce, so a rank-derived key made only the rank
    that saved the cache file hit and deadlocked the others."""
    from flashinfer.fused_moe.cute_dsl.tuner import (
        CuteDslFusedMoERunner,
        CuteDslFusedMoEW4A16Runner,
    )

    runner_cls = {
        "CuteDslFusedMoERunner": CuteDslFusedMoERunner,
        "CuteDslFusedMoEW4A16Runner": CuteDslFusedMoEW4A16Runner,
    }[runner_name]

    def key(**parallel):
        return str(_runner(runner_cls, **parallel).get_cache_key_extras([]))

    # vLLM's flashinfer_cutedsl backend sets offset = ep_rank * num_local_experts.
    assert key(local_expert_offset=0) == key(local_expert_offset=32)
    assert key(local_expert_offset=0) == key(local_expert_offset=64)

    # Configuration that is NOT rank-derived must still distinguish entries.
    assert key(num_experts=256) != key(num_experts=128)
