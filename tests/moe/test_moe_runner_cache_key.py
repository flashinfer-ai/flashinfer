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

from types import SimpleNamespace

import pytest

from flashinfer.fused_moe import (
    BackendOptions,
    CuteDslConfig,
    CuteDslRunner,
    ExpertConfig,
    MoEConfig,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SwiGLU,
    TrtllmFp4Config,
    TrtllmFp4RoutedRunner,
    TrtllmFp8BlockConfig,
    TrtllmFp8BlockRunner,
    TrtllmMxInt4Config,
    TrtllmMxInt4RoutedRunner,
)

_RUNNERS = [
    pytest.param(
        TrtllmFp8BlockRunner,
        TrtllmFp8BlockConfig(),
        QuantConfig(weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8),
        id="trtllm_fp8_block",
    ),
    pytest.param(
        TrtllmFp4RoutedRunner,
        TrtllmFp4Config(),
        QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        id="trtllm_fp4",
    ),
    pytest.param(
        TrtllmMxInt4RoutedRunner,
        TrtllmMxInt4Config(),
        QuantConfig(weight=QuantFormat.MXINT4, activation=QuantFormat.BF16),
        id="trtllm_mxint4",
    ),
    pytest.param(
        CuteDslRunner,
        CuteDslConfig(),
        QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        id="cute_dsl",
    ),
]


def _runner(runner_cls, backend_cfg, quant, **parallel):
    """Build a config-only runner without loading architecture-specific modules."""
    fields = dict(num_experts=256, local_num_experts=32, local_expert_offset=0)
    fields.update(parallel)
    runner = runner_cls.__new__(runner_cls)
    runner.config = MoEConfig(
        routing=RoutingConfig(num_experts=fields["num_experts"], top_k=8),
        quant=quant,
        experts=ExpertConfig(
            intermediate_size=2048,
            local_num_experts=fields["local_num_experts"],
            local_expert_offset=fields["local_expert_offset"],
        ),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=(backend_cfg,)),
    )
    if runner_cls is CuteDslRunner:
        runner._inner = SimpleNamespace(use_fused_finalize=True, enable_pdl=True)
    return runner


@pytest.mark.parametrize("runner_cls,backend_cfg,quant", _RUNNERS)
def test_moe_runner_persisted_cache_key_is_rank_invariant(
    runner_cls, backend_cfg, quant
):
    """Ranks tuned together must share persisted tactics: a cache hit skips the
    set_autotune_process_group reduce, so a rank-derived key made only the rank
    that saved the cache file hit and deadlocked the others (#6082)."""

    def key(**parallel):
        return str(
            _runner(runner_cls, backend_cfg, quant, **parallel).get_cache_key_extras([])
        )

    # EP backends set offset = ep_rank * num_local_experts.
    assert key(local_expert_offset=0) == key(local_expert_offset=32)
    assert key(local_expert_offset=0) == key(local_expert_offset=64)

    # Configuration that is NOT rank-derived must still distinguish entries.
    assert key(num_experts=256) != key(num_experts=128)
    assert key(local_num_experts=32) != key(local_num_experts=64)


@pytest.mark.parametrize("runner_cls,backend_cfg,quant", _RUNNERS)
def test_moe_runner_hash_keeps_local_expert_offset(runner_cls, backend_cfg, quant):
    """The in-memory key still separates shards living in one process."""
    rank0 = _runner(runner_cls, backend_cfg, quant, local_expert_offset=0)
    rank1 = _runner(runner_cls, backend_cfg, quant, local_expert_offset=32)
    same = _runner(runner_cls, backend_cfg, quant, local_expert_offset=0)

    assert hash(rank0) != hash(rank1)
    assert hash(rank0) == hash(same)
