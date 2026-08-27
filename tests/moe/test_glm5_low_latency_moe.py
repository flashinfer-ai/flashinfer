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

"""Tests for the Blackwell GLM5 low-latency MoE path.

The eight-GPU replay uses per-rank tensors dumped from a GLM5 TP8 serving run::

    FLASHINFER_GLM5_LOW_LATENCY_MOE_DUMP_DIR=~/dev/debug_output \
      torchrun --nproc_per_node=8 -m pytest \
      tests/moe/test_glm5_low_latency_moe.py -v -m "gpu_8 and arch_blackwell"
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    Glm5LowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    RoutingInputMode,
    RoutingMethodType,
)
from flashinfer.fused_moe.glm5 import (
    _check_glm5_low_latency_moe_shapes,
    _interleave_packed_gate_up,
    _pack_up_weight_side,
    pack_glm5_low_latency_moe_gate_up_scale,
)

_NUM_EXPERTS = 256
_HIDDEN_SIZE = 6144
_ERROR_THRESHOLDS_BY_RANK = (
    9.30e-5,
    8.30e-5,
    1.23e-4,
    9.50e-5,
    1.00e-4,
    8.10e-5,
    8.90e-5,
    5.50e-4,
)


def _glm5_config(intermediate_size: int) -> MoEConfig:
    return MoEConfig(
        routing=RoutingConfig(
            num_experts=_NUM_EXPERTS,
            top_k=8,
            method=RoutingMethodType.MiniMax2,
            routed_scaling_factor=2.5,
        ),
        quant=QuantConfig(variant=QuantVariant.Glm5LowLatencyFp8),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            num_fused_shared_experts=1,
        ),
        backend=BackendOptions(candidates=(Glm5LowLatencyConfig(),)),
        execution=ExecutionConfig(tune_max_num_tokens=4),
    )


def test_glm5_unified_config_contract() -> None:
    config = _glm5_config(256)
    assert config.backend.valid_for(100) == [Glm5LowLatencyConfig()]
    assert config.backend.valid_for(103) == [Glm5LowLatencyConfig()]
    assert config.backend.valid_for(90) == []
    assert eval(repr(config.backend.candidates[0])) == Glm5LowLatencyConfig()


def test_glm5_up_weight_side_pack_layout() -> None:
    """The side pack must be reversible at byte granularity."""
    logical = (
        (torch.arange(256 * _HIDDEN_SIZE, dtype=torch.int64) % 223 - 111)
        .reshape(256, _HIDDEN_SIZE)
        .to(torch.float8_e4m3fn)
    )
    packed = _pack_up_weight_side(logical)

    logical_shape = (
        4,
        4,
        2,
        8,
        8,
        6,
        4,
        2,
        4,
        4,
    )
    permutation = (0, 4, 5, 1, 6, 3, 8, 7, 2, 9)
    inverse = tuple(permutation.index(dim) for dim in range(len(permutation)))
    unpacked = packed.reshape(4, 8, 6, 4, 4, 8, 4, 2, 2, 4)
    unpacked = unpacked.permute(inverse).reshape(logical_shape).reshape_as(logical)
    assert torch.equal(unpacked.view(torch.uint8), logical.view(torch.uint8))


def test_glm5_gate_up_interleave_layout() -> None:
    gate = torch.zeros((1, 8, 49152), dtype=torch.float8_e4m3fn)
    up = torch.ones_like(gate)
    combined = _interleave_packed_gate_up(gate, up).reshape(
        1, 8, 6, 8, 4, 8, 4, 2, 2, 4
    )
    gate_bytes = combined[..., 0, :].contiguous().view(torch.uint8)
    up_bytes = combined[..., 1, :].contiguous().view(torch.uint8)
    assert torch.count_nonzero(gate_bytes) == 0
    one_byte = up.view(torch.uint8).flatten()[0]
    assert torch.equal(up_bytes, torch.ones_like(up_bytes) * one_byte)


def test_glm5_gate_up_scale_ordering() -> None:
    shared = torch.arange(4 * 48, dtype=torch.float32).reshape(4, 48)
    routed = torch.arange(_NUM_EXPERTS * 4 * 48, dtype=torch.float32).reshape(
        _NUM_EXPERTS, 4, 48
    )
    packed = pack_glm5_low_latency_moe_gate_up_scale(shared, routed)

    assert torch.equal(packed[0], shared)
    assert torch.equal(packed[1:, :2], routed[:, 2:])
    assert torch.equal(packed[1:, 2:], routed[:, :2])


def test_glm5_shape_contract_on_meta_tensors() -> None:
    inter = 256
    args = (
        torch.empty((4, _HIDDEN_SIZE), device="meta", dtype=torch.bfloat16),
        torch.empty((4, _NUM_EXPERTS), device="meta", dtype=torch.float32),
        torch.empty((_NUM_EXPERTS,), device="meta", dtype=torch.bfloat16),
        torch.empty(
            (257, inter // 64, 8, 98304), device="meta", dtype=torch.float8_e4m3fn
        ),
        torch.empty((257, 2 * inter // 128, 48), device="meta", dtype=torch.float32),
        torch.empty(
            (_NUM_EXPERTS, _HIDDEN_SIZE, inter),
            device="meta",
            dtype=torch.float8_e4m3fn,
        ),
        torch.empty(
            (_NUM_EXPERTS, 48, inter // 128), device="meta", dtype=torch.float32
        ),
        torch.empty((_HIDDEN_SIZE, inter), device="meta", dtype=torch.float8_e4m3fn),
        torch.empty((48, inter // 128), device="meta", dtype=torch.float32),
    )
    assert _check_glm5_low_latency_moe_shapes(*args) == (4, inter)

    with pytest.raises(ValueError, match="supports 1 <= M <= 4"):
        _check_glm5_low_latency_moe_shapes(
            torch.empty((5, _HIDDEN_SIZE), device="meta", dtype=torch.bfloat16),
            torch.empty((5, _NUM_EXPERTS), device="meta", dtype=torch.float32),
            *args[2:],
        )


def _single_dump(dump_dir: Path, pattern: str) -> Path:
    matches = sorted(dump_dir.glob(pattern))
    if len(matches) != 1:
        pytest.skip(f"expected one dump matching {pattern!r}, found {len(matches)}")
    return matches[0]


def _load_dump(
    dump_dir: Path, rank: int, layer: int, name: str, device
) -> torch.Tensor:
    path = dump_dir / f"r{rank}_l{layer}_{name}.pt"
    if not path.exists():
        pytest.skip(f"missing GLM5 replay tensor: {path}")
    return torch.load(path, map_location="cpu").to(device)


@pytest.mark.gpu_8
@pytest.mark.arch_blackwell
@pytest.mark.solo
def test_glm5_low_latency_moe_tp8_dump_replay() -> None:
    """Replay all TP8 ranks against saved per-rank PyTorch reference output."""
    dump_env = os.environ.get("FLASHINFER_GLM5_LOW_LATENCY_MOE_DUMP_DIR")
    if not dump_env:
        pytest.skip(
            "set FLASHINFER_GLM5_LOW_LATENCY_MOE_DUMP_DIR to run the GLM5 dump replay"
        )

    if not all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        pytest.skip("run the GLM5 dump replay under torchrun --nproc_per_node=8")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == 8
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    assert torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))

    dump_dir = Path(dump_env).expanduser()
    router_path = _single_dump(dump_dir, f"r{rank}_l*_router_weight.pt")
    hidden_path = _single_dump(dump_dir, f"r{rank}_l*_hidden_states.pt")
    weight_layer = int(router_path.name.split("_", 2)[1][1:])
    activation_layer = int(hidden_path.name.split("_", 2)[1][1:])

    hidden_states = _load_dump(
        dump_dir, rank, activation_layer, "hidden_states", device
    )[:4].contiguous()
    router_weight = _load_dump(dump_dir, rank, weight_layer, "router_weight", device)
    routing_bias = _load_dump(dump_dir, rank, weight_layer, "routing_bias", device)
    router_logits = torch.matmul(
        hidden_states.float(), router_weight.float().transpose(0, 1)
    ).contiguous()

    prepared = Glm5LowLatencyConfig.prepare_weights(
        _load_dump(dump_dir, rank, weight_layer, "shared_gate_up_weight_org", device),
        _load_dump(
            dump_dir,
            rank,
            weight_layer,
            "shared_gate_up_weight_scale_org",
            device,
        ),
        _load_dump(dump_dir, rank, weight_layer, "routed_w3_w1_weight", device),
        _load_dump(
            dump_dir,
            rank,
            weight_layer,
            "routed_w3_w1_weight_scaling_factor",
            device,
        ),
        _load_dump(dump_dir, rank, weight_layer, "routed_w2_weight", device),
        _load_dump(
            dump_dir,
            rank,
            weight_layer,
            "routed_w2_weight_scaling_factor",
            device,
        ),
        _load_dump(dump_dir, rank, weight_layer, "shared_down_weight_org", device),
        _load_dump(
            dump_dir,
            rank,
            weight_layer,
            "shared_down_weight_scale_org",
            device,
        ),
    )
    expected = _load_dump(dump_dir, rank, weight_layer, "pytorch_ref_output", device)[
        :4
    ]

    threshold = _ERROR_THRESHOLDS_BY_RANK[rank]
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for("glm5_low_latency", prepared)
    act_pack = MoEActivationPack(
        hidden_states_q=hidden_states,
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=router_logits,
        routing_bias=routing_bias,
    )
    local_intermediate_size = prepared["shared_down_weight"].shape[1]
    layer = MoELayer(_glm5_config(local_intermediate_size), device=device)
    with torch.inference_mode():
        actual = layer(act_pack, weight_pack)
        torch.cuda.synchronize(device)

    assert local_intermediate_size in (256, 512)
    assert layer.winner_backend == "glm5_low_latency"
    max_abs_error = (actual.float() - expected.float()).abs().max().item()
    print(f"rank={rank} max_abs_error={max_abs_error:.6e} threshold={threshold:.6e}")
    assert max_abs_error <= threshold


def test_glm5_tp8_dump_replay_requires_torchrun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FLASHINFER_GLM5_LOW_LATENCY_MOE_DUMP_DIR", str(tmp_path))
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(
        pytest.skip.Exception,
        match="run the GLM5 dump replay under torchrun --nproc_per_node=8",
    ):
        test_glm5_low_latency_moe_tp8_dump_replay()
