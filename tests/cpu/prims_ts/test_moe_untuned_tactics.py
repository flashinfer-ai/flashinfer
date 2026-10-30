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

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
from flashinfer.prims_ts.moe.config_mapper import (
    SUPPORTED_BF16_TILE_N,
    SUPPORTED_NVFP4_TILE_N,
    _default_tile_n,
)
from flashinfer.prims_ts.moe.runner import untuned_moe_tactics


@dataclass(frozen=True)
class _TileRunner:
    """Resolves tactic -1 like a Prims-TS runner, keeping only the tile."""

    top_k: int
    num_experts: int
    num_local_experts: int
    supported_tiles: tuple[int, ...]

    def _map_tactic(self, tactic, moe_inputs, kwargs):
        assert tactic == [-1, -1]
        tile_n = _untuned_tile(self, moe_inputs.hidden_states.shape[0])
        return SimpleNamespace(tile_n=tile_n, moe_config_index=0)


def _untuned_tile(runner: _TileRunner, num_tokens: int) -> int:
    return _default_tile_n(
        num_tokens=num_tokens,
        top_k=runner.top_k,
        num_local_experts=runner.num_local_experts,
        num_experts=runner.num_experts,
        supported_tiles=runner.supported_tiles,
    )


def _moe_inputs(num_tokens: int) -> MoeRunnerInputs:
    return MoeRunnerInputs(
        output=torch.empty(0),
        routing_logits=None,
        topk_ids=None,
        expert_weights=None,
        hidden_states=torch.empty((num_tokens, 8)),
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "runner",
    [
        _TileRunner(4, 128, 32, SUPPORTED_NVFP4_TILE_N),
        _TileRunner(4, 128, 128, SUPPORTED_NVFP4_TILE_N),
        _TileRunner(8, 64, 64, SUPPORTED_NVFP4_TILE_N),
        # Tokens per expert is not a power-of-two fraction of the token count.
        _TileRunner(6, 40, 40, SUPPORTED_NVFP4_TILE_N),
        _TileRunner(1, 1, 1, SUPPORTED_NVFP4_TILE_N),
        _TileRunner(8, 256, 32, SUPPORTED_BF16_TILE_N),
    ],
)
def test_untuned_tactics_cover_every_token_count(runner):
    """Every token count's untuned tile is among the probed tactics, and only those."""
    tactics = untuned_moe_tactics(runner, _moe_inputs(3), {})

    expected = {(_untuned_tile(runner, n), 0) for n in range(1, 1 << 14)}
    assert {tuple(tactic) for tactic in tactics} == expected


@pytest.mark.cpu_only
def test_untuned_tactics_probe_without_copying_inputs():
    seen = []

    class _RecordingRunner:
        def _map_tactic(self, tactic, moe_inputs, kwargs):
            seen.append((moe_inputs.hidden_states, kwargs))
            return SimpleNamespace(tile_n=8, moe_config_index=0)

    kwargs = {"enable_pdl": True}
    moe_inputs = _moe_inputs(3)

    assert untuned_moe_tactics(_RecordingRunner(), moe_inputs, kwargs) == ([8, 0],)
    assert all(passed is kwargs for _, passed in seen)
    for hidden_states, _ in seen:
        assert hidden_states.data_ptr() == moe_inputs.hidden_states.data_ptr()
        assert hidden_states.shape[1:] == moe_inputs.hidden_states.shape[1:]
