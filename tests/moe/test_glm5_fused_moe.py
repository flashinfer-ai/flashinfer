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

"""Tests for the Blackwell GLM5 low-token fused MoE path.

The eight-GPU replay uses per-rank tensors dumped from a GLM5 TP8 serving run::

    FLASHINFER_GLM5_MOE_DUMP_DIR=~/dev/debug_output \
      torchrun --nproc_per_node=8 -m pytest \
      tests/moe/test_glm5_fused_moe.py -v -m "gpu_8 and arch_blackwell"
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from flashinfer.fused_moe.glm5 import (
    _check_glm5_fused_moe_shapes,
    _interleave_packed_gate_up,
    _pack_up_weight_side,
    alloc_glm5_fused_moe_workspace,
    glm5_fused_moe,
    pack_glm5_fused_moe_gate_up_scale,
    prepare_glm5_fused_moe_weights,
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
    packed = pack_glm5_fused_moe_gate_up_scale(shared, routed)

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
    assert _check_glm5_fused_moe_shapes(*args) == (4, inter)

    with pytest.raises(ValueError, match="supports 1 <= M <= 4"):
        _check_glm5_fused_moe_shapes(
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
def test_glm5_fused_moe_tp8_dump_replay() -> None:
    """Replay all TP8 ranks against saved per-rank PyTorch reference output."""
    dump_env = os.environ.get("FLASHINFER_GLM5_MOE_DUMP_DIR")
    if not dump_env:
        pytest.skip("set FLASHINFER_GLM5_MOE_DUMP_DIR to run the GLM5 dump replay")

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

    prepared = prepare_glm5_fused_moe_weights(
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
    workspace = alloc_glm5_fused_moe_workspace(
        hidden_states.shape[0], prepared.shared_down_weight.shape[1], device
    )
    output = torch.empty_like(hidden_states)
    for packed_weight_stages, use_tma in (
        (1, True),
        (2, True),
        (1, False),
        (2, False),
    ):
        with torch.inference_mode():
            actual = glm5_fused_moe(
                hidden_states,
                router_logits,
                routing_bias,
                **prepared.as_kwargs(),
                out=output,
                workspace=workspace,
                packed_weight_stages=packed_weight_stages,
                use_tma=use_tma,
            )
            torch.cuda.synchronize(device)

        max_abs_error = (actual.float() - expected.float()).abs().max().item()
        print(
            f"rank={rank} stages={packed_weight_stages} tma={int(use_tma)} "
            f"max_abs_error={max_abs_error:.6e} threshold={threshold:.6e}"
        )
        assert max_abs_error <= threshold
