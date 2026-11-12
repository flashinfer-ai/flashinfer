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

from unittest.mock import MagicMock

import pytest
import torch

import flashinfer.fused_moe.core as core


@pytest.fixture
def moe_runner_cls(monkeypatch):
    """The CUTLASS MoERunner class without building the JIT module."""
    core.get_cutlass_fused_moe_module.cache_clear()
    monkeypatch.setattr(
        core, "gen_cutlass_fused_moe_sm100_module", lambda *_: MagicMock()
    )
    try:
        yield core.get_cutlass_fused_moe_module("100").MoERunner
    finally:
        core.get_cutlass_fused_moe_module.cache_clear()


def _runner(runner_cls, **parallel):
    kwargs = dict(
        x_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        top_k=8,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        cluster_size=1,
        cluster_rank=0,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        enable_pdl=False,
        activation_type=core.ActivationType.Swiglu,
        use_packed_weights=False,
        use_fused_finalize=True,
        use_wfp4afp8_humming=False,
    )
    kwargs.update(parallel)
    return runner_cls(**kwargs)


@pytest.mark.parametrize(
    "size,rank",
    [("tp_size", "tp_rank"), ("ep_size", "ep_rank"), ("cluster_size", "cluster_rank")],
)
def test_cutlass_moe_cache_key_is_rank_invariant(moe_runner_cls, size, rank):
    """Ranks tuned together must share persisted tactics: a cache hit skips the
    set_autotune_process_group reduce, so a rank-specific key made only the rank
    that saved the cache file hit and deadlocked the others."""

    def key(**parallel):
        return str(_runner(moe_runner_cls, **parallel).get_cache_key_extras([]))

    assert key(**{size: 2, rank: 0}) == key(**{size: 2, rank: 1})
    assert key(**{size: 2, rank: 0}) != key(**{size: 4, rank: 0})
