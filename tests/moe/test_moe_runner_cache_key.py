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

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    CuteDslConfig,
    CuteDslRunner,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
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


_HIDDEN = 256
_INTERMEDIATE = 256
_NUM_EXPERTS = 16
_TOP_K = 2
_TOKENS = 64
_FP8_BLOCK = QuantConfig(
    weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8
)


def _trtllm_fp8_block_supported() -> bool:
    from flashinfer.fused_moe.api import _TRTLLM_ROUTED_FP8_ARCHS
    from flashinfer.utils import get_compute_capability

    if not torch.cuda.is_available():
        return False
    major, minor = get_compute_capability(torch.device("cuda"))
    return major * 10 + minor in _TRTLLM_ROUTED_FP8_ARCHS


def _trtllm_fp8_block_layer(local_expert_offset: int, local_num_experts: int):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(7920)
    bf16 = dict(device=device, dtype=torch.bfloat16, generator=gen)
    x = torch.randn(_TOKENS, _HIDDEN, **bf16)
    w1 = torch.randn(local_num_experts, 2 * _INTERMEDIATE, _HIDDEN, **bf16) * 0.02
    w2 = torch.randn(local_num_experts, _HIDDEN, _INTERMEDIATE, **bf16) * 0.02
    ids = torch.randint(
        local_expert_offset,
        local_expert_offset + local_num_experts,
        (_TOKENS, _TOP_K),
        device=device,
        dtype=torch.int32,
        generator=gen,
    )
    route_weights = torch.softmax(
        torch.randn(_TOKENS, _TOP_K, device=device, generator=gen), dim=-1
    )
    x_q, x_scale = TrtllmFp8BlockConfig.prepare_activations(x, quant=_FP8_BLOCK)
    view = TrtllmFp8BlockConfig.prepare_weights(
        w1,
        w2,
        quant=_FP8_BLOCK,
        num_local_experts=local_num_experts,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        activation=SwiGLU(),
        device=device,
    )
    weights = MoEWeightPack()
    weights.prepare_for("trtllm_fp8_block", view)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=_NUM_EXPERTS, top_k=_TOP_K),
        quant=_FP8_BLOCK,
        experts=ExpertConfig(
            intermediate_size=_INTERMEDIATE,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
        ),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=(TrtllmFp8BlockConfig(),)),
        execution=ExecutionConfig(tune_max_num_tokens=_TOKENS),
    )
    act = MoEActivationPack(
        hidden_states_q=x_q,
        hidden_states_scale=x_scale,
        topk_ids=ids,
        topk_weights=route_weights,
    )
    return MoELayer(config), act, weights


def _cross_process_worker(cache: str, local_expert_offset: int, local_num_experts: int):
    """Tune one layer against ``cache`` and report how many tactics were measured."""
    from flashinfer.autotuner import AutoTuner, autotune

    measured = 0

    def counting(method):
        def wrapped(self, *args, **kwargs):
            nonlocal measured
            measured += 1
            return method(self, *args, **kwargs)

        return wrapped

    for name in ("_profile_single_kernel", "_profile_single_kernel_cupti"):
        setattr(AutoTuner, name, counting(getattr(AutoTuner, name)))

    layer, act, weights = _trtllm_fp8_block_layer(
        local_expert_offset, local_num_experts
    )
    with autotune(True, cache=cache):
        out = layer(act, weights)
    torch.cuda.synchronize()
    print(
        "RESULT "
        + json.dumps({"measured": measured, "finite": bool(out.isfinite().all())})
    )


def _run_worker(cache, local_expert_offset: int, local_num_experts: int) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            str(cache),
            str(local_expert_offset),
            str(local_num_experts),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert result, proc.stdout[-2000:]
    return json.loads(result[-1][len("RESULT ") :])


def _persisted_entries(cache) -> dict:
    configs = json.loads(cache.read_text())
    return {k: v for k, v in configs.items() if not k.startswith("_")}


@pytest.mark.skipif(
    not _trtllm_fp8_block_supported(),
    reason="TRT-LLM block-FP8 MoE requires an arch in _TRTLLM_ROUTED_FP8_ARCHS",
)
def test_moe_runner_persisted_cache_hits_across_processes_and_ranks(tmp_path):
    """A warm restart on another EP rank loads the persisted tactic instead of re-tuning (#6082)."""
    cache = tmp_path / "autotune_cache.json"

    rank0 = _run_worker(cache, local_expert_offset=0, local_num_experts=8)
    assert rank0["measured"] > 0 and rank0["finite"]
    tuned = _persisted_entries(cache)
    assert any("moe_trtllm_fp8_block" in key for key in tuned), tuned

    rank1 = _run_worker(cache, local_expert_offset=8, local_num_experts=8)
    assert rank1["measured"] == 0 and rank1["finite"]
    assert _persisted_entries(cache) == tuned

    # A configuration that is not rank-derived still gets its own entry.
    other = _run_worker(cache, local_expert_offset=0, local_num_experts=16)
    assert other["measured"] > 0
    assert set(_persisted_entries(cache)) > set(tuned)


if __name__ == "__main__":
    _cross_process_worker(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
